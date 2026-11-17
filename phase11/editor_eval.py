"""Phase 11 editor evaluation pass — counter-enforced, ledger-pre-logged.

Runs INSIDE WSL. Wraps eval_gemma.py: checks the pass budget (10), appends the
hypothesis to the ledger BEFORE running, evaluates 64 practice problems on the
sandbox model, prints the summary JSON.

Usage:
  ~/wsp3c-venv/bin/python editor_eval.py --name ed_pass1 \
      --hypothesis "baseline: unmodified model, observe failure mode"
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDITORD = HERE / "editor"
PASSES = EDITORD / "passes.txt"
LEDGER = EDITORD / "ledger.md"
MAX_PASSES = 10


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
    used = int(passes_file.read_text()) if passes_file.exists() else 0
    if used >= MAX_PASSES:
        print(json.dumps({"refused": True, "reason": "budget exhausted",
                          "used": used, "max": MAX_PASSES}))
        sys.exit(2)

    used += 1
    passes_file.write_text(str(used))
    if used == 1 and not ledger_file.exists():
        ledger_file.write_text("| pass | time | hypothesis |\n|---|---|---|\n")
    with ledger_file.open("a") as f:
        f.write(f"| {used} | {datetime.now().isoformat(timespec='seconds')} | {args.hypothesis} |\n")

    cmd = [sys.executable, str(HERE / "eval_gemma.py"),
           "--model", str(Path.home() / "wsp11/sandbox/model"),
           "--name", args.name, "--n", "64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = outdir / f"{args.name}.log"
    log.write_text(r.stdout + r.stderr)
    if (HERE / f"{args.name}.json").exists():
        summary = json.loads((HERE / f"{args.name}.json").read_text())["summary"]
        summary["pass"] = used
        summary["remaining"] = MAX_PASSES - used
        print(json.dumps(summary))
    else:
        print(json.dumps({"failed": True, "log": str(log),
                          "stderr_tail": r.stderr[-400:]}))
        sys.exit(1)


if __name__ == "__main__":
    main()
