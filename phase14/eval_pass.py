"""Phase-14 forward-pass tool: the only way to measure a candidate (budget- and rule-enforced).

One pass runs the first --n practice problems (default: all 128) under the measurement protocol of
phase14/protocol.md — the reference benchmark question, temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, at most 32,768 new tokens, one sample, a fixed seed and the same problems
every pass — and prints a summary JSON. The pass is charged before the engine starts and refunded
if the engine fails. A candidate must carry a receipt from save_edit.py, or be the unmodified base.

Lives in the sandbox as harness/eval_pass.py.

Usage (inside WSL): python eval_pass.py <candidate_model_dir> --hypothesis "one sentence" [--n 128]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent          # the sandbox's harness/ folder
sys.path.insert(0, str(HERE))
from gpulock import wait_for_gpu  # noqa: E402
from rescore import answers_match, extract_answer  # noqa: E402
from runcheck import check_run  # noqa: E402
from save_edit import fingerprint  # noqa: E402

PRACTICE = HERE.parent / "practice.json"
STATE = HERE / "state"
LEDGER, COUNTER, RECEIPTS = STATE / "ledger.md", STATE / "passes.txt", STATE / "receipts"
MAX_PASSES, CAP, SEED = 15, 32768, 20261124
# the benchmark question as the original authors' evaluation put it (phase3c/run_vllm.py has the same constant)
REFERENCE_QUESTION = ("Problem:\n{q}\n\nPlease reason step by step, and put your final answer "
                      "within \\boxed{{}}.")
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def used():
    return int(COUNTER.read_text()) if COUNTER.exists() else 0


def log(line):
    with LEDGER.open("a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--hypothesis", required=True, help="one sentence: what you predict and why")
    ap.add_argument("--n", type=int, default=128, help="practice problems to run (at most 128)")
    args = ap.parse_args()
    cand = Path(args.candidate).expanduser()
    STATE.mkdir(exist_ok=True)
    k = used()
    if k >= MAX_PASSES:
        print(json.dumps(dict(refused=True, reason="budget exhausted", used=k, max=MAX_PASSES)))
        sys.exit(2)

    fp = fingerprint(cand)
    receipt = RECEIPTS / f"{fp}.json"
    is_base = fp == (HERE / "base_fingerprint.txt").read_text().strip()
    if not is_base and not receipt.exists():
        print(json.dumps(dict(refused=True, reason="no receipt: run save_edit.py on this candidate first",
                              fingerprint=fp[:16])))
        sys.exit(2)
    practice = json.loads(PRACTICE.read_text(encoding="utf-8"))[: min(args.n, 128)]

    k += 1
    COUNTER.write_text(str(k))
    log(f"| {k} | {time.strftime('%Y-%m-%dT%H:%M:%S')} | {'base model' if is_base else fp[:16]} "
        f"| n={len(practice)} | {args.hypothesis} |")
    t0 = time.time()
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
        wait_for_gpu()
        llm = LLM(model=str(cand), max_model_len=40000, gpu_memory_utilization=0.92,
                  enforce_eager=False, enable_prefix_caching=False, max_num_seqs=64)
        outs = llm.generate(prompts, sp)
    except Exception as e:  # the engine failed: the pass is not spent (original rule)
        COUNTER.write_text(str(k - 1))
        log(f"| {k} | {time.strftime('%Y-%m-%dT%H:%M:%S')} | FAILED, pass refunded | {type(e).__name__}: "
            f"{str(e)[-300:].replace(chr(10), ' ')} |")
        print(json.dumps(dict(failed=True, refunded=True, error=f"{type(e).__name__}: {str(e)[-300:]}")))
        sys.exit(1)

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
    failed, validity = check_run(rows, texts, ok_key="ok_boxed")

    n = len(rows)
    pct = lambda f: round(100 * sum(f(r) for r in rows) / n, 1)   # noqa: E731
    by_level = {}
    for p, r in zip(practice, rows):
        correct, total = by_level.get(p["level"], [0, 0])
        by_level[p["level"]] = [correct + int(r["ok_boxed"]), total + 1]
    summary = dict(pass=k, remaining=MAX_PASSES - k, n=n,
                   accuracy_boxed=pct(lambda r: r["ok_boxed"]),
                   accuracy_rescore=pct(lambda r: r["ok_rescored"]),
                   closed=pct(lambda r: r["closed"]), cap_hit_rate=pct(lambda r: r["finish"] == "length"),
                   mean_tokens=round(sum(r["n_tokens"] for r in rows) / n),
                   by_level=by_level,
                   candidate="base model" if is_base else fp[:16], wall_s=round(time.time() - t0, 1),
                   validity=validity, failed_checks=failed)
    with (STATE / f"gen_{k}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (STATE / "eval_pass_last.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if failed:
        print("RUN VALIDITY FAILED (ISSUES #20); files kept as evidence: " + "; ".join(failed), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
