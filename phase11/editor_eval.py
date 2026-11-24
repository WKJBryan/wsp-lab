"""Phase 11 editor evaluation pass — counter-enforced, ledger-pre-logged.

Runs INSIDE WSL. Wraps eval_gemma.py: checks the pass budget (10), appends the
hypothesis to the ledger BEFORE running, evaluates 64 practice problems on the
sandbox model, prints the summary JSON. A result file of the same name must not
already exist (checked before a pass is charged). A pass whose evaluation fails
for any reason — engine error, or a failed validity check — is refunded.
A wrapper killed while its evaluator runs cannot refund: that pass stays charged
and its ledger row has no outcome (the safe direction for a budget).

Usage:
  ~/wsp3c-venv/bin/python editor_eval.py --name ed_pass1 \
      --hypothesis "baseline: unmodified model, observe failure mode"
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDITORD = HERE / "editor"
MAX_PASSES = 10


def cell(text):
    """One ledger cell: a pipe or a line break inside it would split the row."""
    return str(text).replace("|", "/").replace("\r", " ").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--outdir", default=str(EDITORD),
                    help="editor session dir (passes/ledger/logs); Am.#3: editor_flash for the GLM-5.3-Flash rerun")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    passes_file = outdir / "passes.txt"
    ledger_file = outdir / "ledger.md"
    result = outdir / f"{args.name}.json"
    if ledger_file.exists() and not passes_file.exists():
        sys.exit(f"{ledger_file} exists without {passes_file.name}: the pass count would restart at zero")
    if result.exists() or (outdir / f"{args.name}_texts.json").exists():
        sys.exit(f"{result} already exists: the evaluator would refuse it and the pass would be spent "
                "for nothing; choose a new --name")
    try:
        used = int(passes_file.read_text()) if passes_file.exists() else 0
    except ValueError:     # left empty by a crash: nothing was charged
        used = 0
    if used >= MAX_PASSES:
        print(json.dumps({"refused": True, "reason": "budget exhausted", "used": used, "max": MAX_PASSES}))
        sys.exit(2)

    used += 1
    passes_file.write_text(str(used))
    if used == 1 and not ledger_file.exists():
        ledger_file.write_text("| pass | time | hypothesis |\n|---|---|---|\n")
    with ledger_file.open("a") as f:
        f.write(f"| {used} | {datetime.now().isoformat(timespec='seconds')} | {cell(args.hypothesis)} |\n")

    # the result goes to this session's folder, never the shared top level: earlier attempts
    # overwrote each other's ed_pass*.json there (ISSUES #20, #21)
    cmd = [sys.executable, str(HERE / "eval_gemma.py"),
           "--model", str(Path.home() / "wsp11/sandbox/model"),
           "--name", args.name, "--n", "64", "--outdir", str(outdir)]
    started = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = outdir / f"{args.name}.log"
    log.write_text(r.stdout + r.stderr)
    # success = a result file written during this call whose run passed its validity check; the
    # evaluator's exit status is recorded but does not decide (an engine can die at teardown after
    # a clean run, and exit 2 after a failed check leaves a file that says so). An older file of
    # the same name must never be reported as the outcome of the edit just tested.
    measured = None
    if result.exists() and result.stat().st_mtime >= started - 1:
        summary = json.loads(result.read_text())["summary"]
        if not (summary.get("validity") or {}).get("problems"):
            measured = summary
    if measured is not None:
        measured["pass"] = used
        measured["remaining"] = MAX_PASSES - used
        if r.returncode != 0:
            measured["evaluator_exit"] = r.returncode
            with ledger_file.open("a") as f:
                f.write(f"| {used} | {datetime.now().isoformat(timespec='seconds')} | MEASURED (evaluator exit "
                        f"{r.returncode} after the result was written) | see {log.name} |\n")
        print(json.dumps(measured))
    else:
        # no measurement came out of it, so it is not spent (the original authors' rule)
        why = f"evaluator exit {r.returncode}"
        if result.exists() and result.stat().st_mtime >= started - 1:
            found = (json.loads(result.read_text())["summary"].get("validity") or {}).get("problems") or []
            if found:
                why = "run validity: " + "; ".join(p.replace("|", "/") for p in found)
        passes_file.write_text(str(used - 1))
        with ledger_file.open("a") as f:
            f.write(f"| {used} | {datetime.now().isoformat(timespec='seconds')} | FAILED, pass refunded "
                    f"| {why} | see {log.name} |\n")
        print(json.dumps({"failed": True, "refunded": True, "log": str(log), "exit_code": r.returncode,
                          "stderr_tail": r.stderr[-400:]}))
        sys.exit(1)


if __name__ == "__main__":
    main()
