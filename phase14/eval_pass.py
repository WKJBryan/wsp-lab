"""Phase-14 forward-pass tool: the only way to measure a candidate (budget- and rule-enforced).

One pass runs the first --n practice problems (default: all 128) under the measurement protocol of
phase14/protocol.md — the reference benchmark question, temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, at most 32,768 new tokens, one sample, a fixed seed and the same problems
every pass — and prints a summary JSON. A candidate must carry a receipt from save_edit.py, or be
the unmodified base.

Budget. The tool waits for the project's GPU lock first, then charges the pass; so a tool that is
killed while it waits has charged nothing, and two tools cannot charge against each other. A pass
is refunded when the engine fails, when the tool is interrupted (Ctrl-C or a termination request),
and when the run fails the batch-position check of runcheck.py (an outcome that depends on the
request's position is the engine's doing, not a property of the candidate: ISSUES #20). A run whose
generations end in fragments is charged: fragments are the candidate's own output, and the summary
says so. Every attempt has its own number, refunded ones included: its texts are kept as
state/gen_<attempt>.jsonl from the moment generation returns, and the ledger gets one row for the
hypothesis and one for the outcome (MEASURED, FAILED or INTERRUPTED). Only a tool killed with
SIGKILL leaves a hypothesis row without an outcome row: that pass is spent.

The candidate is fingerprinted twice: once before the lock (a candidate without a receipt is
refused before anything is charged) and again under the lock just before the engine loads it. The
fingerprint covers every file of the directory, so a candidate that changed while it waited for
the GPU — weights, config or chat template — is refused and the pass refunded. Before anything
else the tool checks that its own files are the ones the sandbox was built with.

Lives in the sandbox as harness/eval_pass.py.

Usage (inside WSL): python eval_pass.py <candidate_model_dir> --hypothesis "one sentence" [--n 128]
"""

import argparse
import hashlib
import json
import re
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent          # the sandbox's harness/ folder
sys.path.insert(0, str(HERE))
from benchmark import REFERENCE_QUESTION  # noqa: E402
from gpulock import wait_for_gpu  # noqa: E402
from rescore import answers_match, extract_answer  # noqa: E402
from runcheck import check_run  # noqa: E402
from save_edit import fingerprint  # noqa: E402

PRACTICE = HERE.parent / "practice.json"
STATE = HERE / "state"
LEDGER, COUNTER, ATTEMPTS, RECEIPTS = (STATE / "ledger.md", STATE / "passes.txt", STATE / "attempts.txt",
                                       STATE / "receipts")
MAX_PASSES, CAP, SEED = 15, 32768, 20261124
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def counter_value(path):
    """The counter in `path`: 0 when it was never written, or a crash left it empty."""
    try:
        return int(path.read_text())
    except (FileNotFoundError, ValueError):
        return 0


def write_counter(path, value):
    """Replace a counter file only once the new value is complete on disk."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(str(value))
    tmp.replace(path)


def cell(text):
    """One ledger cell: a pipe or a line break inside it would split the row."""
    return str(text).replace("|", "/").replace("\r", " ").replace("\n", " ")


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(line):
    with LEDGER.open("a") as f:
        f.write(line + "\n")


def verify_harness():
    """The tools must be the ones the sandbox was built with: digests recorded by make_sandbox.sh.
    Returns the digests, for the summary."""
    recorded = STATE / "harness.sha256.json"
    if not recorded.exists():
        sys.exit(f"{recorded} is missing: the sandbox was not built with make_sandbox.sh")
    digests = json.loads(recorded.read_text(encoding="utf-8"))
    present = sorted(p.name for p in HERE.glob("*.py"))
    if present != sorted(digests):
        sys.exit(f"harness files {present} differ from the recorded ones {sorted(digests)}")
    for name, digest in digests.items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            sys.exit(f"{name} does not match the digest recorded for the sandbox")
    return digests


def read_outputs(practice, outs):
    """One row per practice problem, and the generations by id; nothing is silently truncated."""
    if len(outs) != len(practice):
        raise RuntimeError(f"the engine returned {len(outs)} outputs for {len(practice)} problems")
    rows, texts = [], {}
    for p, o in zip(practice, outs):
        c = o.outputs[0]
        text, n_tok, finish = c.text, len(c.token_ids), c.finish_reason
        boxed = (BOXED.findall(text) or [None])[-1]
        pred = extract_answer(text)
        rows.append(dict(id=p["id"], gold=p["gold"], text=text,
                         pred_boxed=boxed, ok_boxed=boxed is not None and answers_match(p["gold"], boxed),
                         pred_rescore=pred, ok_rescored=answers_match(p["gold"], pred),
                         n_tokens=n_tok, finish=finish, closed="</think>" in text))
        texts[p["id"]] = text
    return rows, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--hypothesis", required=True, help="one sentence: what you predict and why")
    ap.add_argument("--n", type=int, default=128, help="practice problems to run (at most 128)")
    args = ap.parse_args()
    if not 1 <= args.n <= 128:
        sys.exit(f"--n must be between 1 and 128 (got {args.n})")
    cand = Path(args.candidate).expanduser()
    if not cand.is_dir():
        sys.exit(f"{cand} is not a model directory")
    STATE.mkdir(exist_ok=True)
    digests = verify_harness()
    harness_digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:16]

    base_file = STATE / "base_fingerprint.txt"
    if not base_file.exists():
        sys.exit(f"{base_file} is missing: the sandbox was not built with make_sandbox.sh")
    base_fp = base_file.read_text().strip()
    fp = fingerprint(cand)
    receipt = RECEIPTS / f"{fp}.json"
    is_base = fp == base_fp
    if not is_base and not receipt.exists():
        print(json.dumps(dict(refused=True, reason="no receipt: run save_edit.py on this candidate first",
                              fingerprint=fp[:16])))
        sys.exit(2)
    practice = json.loads(PRACTICE.read_text(encoding="utf-8"))[: args.n]

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(str(cand))
        prompts = [tok.apply_chat_template([{"role": "user", "content": REFERENCE_QUESTION.format(q=p["q"])}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=True)
                   for p in practice]
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
                            repetition_penalty=1.0, max_tokens=CAP, seed=SEED)
    except Exception as e:   # nothing has been charged yet
        print(json.dumps(dict(failed=True, charged=False, error=cell(f"{type(e).__name__}: {str(e)[-300:]}"))))
        sys.exit(1)

    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))   # a kill request refunds the pass
    wait_for_gpu()                 # the lock is held from here until this process ends

    k = counter_value(COUNTER)
    if k >= MAX_PASSES:
        print(json.dumps(dict(refused=True, reason="budget exhausted", used=k, max=MAX_PASSES)))
        sys.exit(2)
    attempt = counter_value(ATTEMPTS) + 1
    k += 1
    write_counter(COUNTER, k)
    write_counter(ATTEMPTS, attempt)
    log(f"| {attempt} | {stamp()} | {'base model' if is_base else fp[:16]} | n={len(practice)} | "
        f"{cell(args.hypothesis)} |")
    t0 = time.time()

    def refund(outcome):
        """Give the pass back and write the outcome row (`outcome` is already ledger-safe)."""
        write_counter(COUNTER, k - 1)
        log(f"| {attempt} | {stamp()} | {outcome} |")

    try:
        if fingerprint(cand) != fp:
            raise RuntimeError("the candidate changed while the pass waited for the GPU")
        llm = LLM(model=str(cand), max_model_len=40000, gpu_memory_utilization=0.92,
                  enforce_eager=False, enable_prefix_caching=False)
        outs = llm.generate(prompts, sp)
        rows, texts = read_outputs(practice, outs)
        with (STATE / f"gen_{attempt}.jsonl").open("w", encoding="utf-8") as f:   # the texts exist from here on
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    except Exception as e:   # the engine failed: the pass is not spent (original rule)
        error = f"{type(e).__name__}: {str(e)[-300:]}"
        refund(f"FAILED, pass refunded | {cell(error)}")
        print(json.dumps(dict(failed=True, refunded=True, remaining=MAX_PASSES - (k - 1), error=error)))
        sys.exit(1)
    except BaseException:    # interrupted (Ctrl-C, a termination request): no measurement came out
        refund("INTERRUPTED, pass refunded")
        raise

    try:
        failed, validity = check_run(rows, texts, ok_key="ok_boxed")
        n = len(rows)
        pct = lambda f: round(100 * sum(f(r) for r in rows) / n, 1)   # noqa: E731
        by_level = {}
        for p, r in zip(practice, rows):
            correct, total = by_level.get(p["level"], [0, 0])
            by_level[p["level"]] = [correct + int(r["ok_boxed"]), total + 1]
        summary = dict(remaining=MAX_PASSES - k, passes_spent=k, n=n,      # "pass" is a keyword: added below
                       accuracy_boxed=pct(lambda r: r["ok_boxed"]),
                       accuracy_rescore=pct(lambda r: r["ok_rescored"]),
                       closed=pct(lambda r: r["closed"]), cap_hit_rate=pct(lambda r: r["finish"] == "length"),
                       mean_tokens=round(sum(r["n_tokens"] for r in rows) / n),
                       by_level=by_level,
                       candidate="base model" if is_base else fp[:16], wall_s=round(time.time() - t0, 1),
                       texts=f"gen_{attempt}.jsonl", harness_digest=harness_digest, base_fingerprint=base_fp,
                       validity=validity, failed_checks=failed)
        summary["pass"] = attempt
    except Exception as e:   # the run is on disk; the scoring is what failed
        error = f"{type(e).__name__}: {str(e)[-300:]}"
        refund(f"FAILED after generation, pass refunded | texts kept in gen_{attempt}.jsonl | {cell(error)}")
        print(json.dumps(dict(failed=True, refunded=True, remaining=MAX_PASSES - (k - 1),
                              texts=f"gen_{attempt}.jsonl", error=error)))
        sys.exit(1)
    (STATE / f"eval_pass_{attempt}.json").write_text(json.dumps(summary, indent=1))

    if any(p.startswith("admission order") for p in failed):
        # not a measurement of the candidate (ISSUES #20): the summary stays in the file for the
        # audit and is not reported as a result
        refund("FAILED, pass refunded | run validity: " + cell("; ".join(failed)))
        print(json.dumps(dict(failed=True, refunded=True, remaining=MAX_PASSES - (k - 1),
                              checks=failed, texts=f"gen_{attempt}.jsonl")))
        sys.exit(2)
    tails = [p for p in failed if p.startswith("soup")]
    log(f"| {attempt} | {stamp()} | MEASURED | accuracy_boxed {summary['accuracy_boxed']}, "
        f"cap-hit {summary['cap_hit_rate']}"
        + (f" | {cell('; '.join(tails))}: the candidate's own output, the pass counts |" if tails else " |"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
