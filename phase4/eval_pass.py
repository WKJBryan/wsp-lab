"""Phase-4 harness-side practice evaluation (runs INSIDE WSL, from the sandbox).

64 fixed GSM8K-train problems, thinking on, temp 0.6 / top-p 0.95 / top-k 20 /
presence penalty 1.5, cap 8192, 1 sample. Prints one JSON line: summary + 3 sample tails.

Budget is HARNESS-ENFORCED: a counter file in the cwd allows at most 10 invocations;
each requires --hypothesis, which is appended to the ledger BEFORE the run.

Usage: $HOME/wsp3c-venv/bin/python eval_pass.py <model_dir> <problems.json> --hypothesis "..."
"""

import datetime
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

CAP = 8192
SEED = 20261105
MAX_PASSES = 10


def spend_pass(hypothesis: str) -> int:
    cnt_file = "eval_counter.txt"
    n = 0
    if os.path.exists(cnt_file):
        n = int(open(cnt_file).read().strip() or 0)
    if n >= MAX_PASSES:
        print(json.dumps(dict(error=f"budget exhausted: {n}/{MAX_PASSES} passes spent")))
        sys.exit(1)
    open(cnt_file, "w").write(str(n + 1))
    with open("ledger.md", "a") as f:
        f.write(
            f"| {n + 1} | {datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"{hypothesis.replace('|', '/')} |\n"
        )
    return n + 1


def main():
    model_dir = sys.argv[1]
    problems_path = sys.argv[2]
    hypothesis = None
    if "--hypothesis" in sys.argv:
        hypothesis = sys.argv[sys.argv.index("--hypothesis") + 1]
    if not hypothesis:
        print(json.dumps(dict(error="--hypothesis is required (pre-logging rule)")))
        sys.exit(1)
    pass_index = spend_pass(hypothesis)
    problems = json.loads(open(problems_path).read())

    llm = LLM(model=model_dir, max_model_len=12000, gpu_memory_utilization=0.92)
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        max_tokens=CAP, seed=SEED + pass_index,
    )
    msgs = [[{"role": "user", "content": p["q"]}] for p in problems]
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})

    import re
    n, capped, closed = len(problems), 0, 0
    samples = []
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        finish = o.outputs[0].finish_reason
        n_tok = len(o.outputs[0].token_ids)
        capped += finish == "length"
        closed += "</think>" in text
        if finish == "length" and len(samples) < 3:
            samples.append(dict(id=p["id"], n_tokens=n_tok,
                                tail=text[-300:], head=text[:120]))
    print(json.dumps(dict(
        n=n, pass_index=pass_index,
        cap_hit_rate=round(capped / n * 100, 1),
        closed_rate=round(closed / n * 100, 1),
        mean_tokens=round(sum(len(o.outputs[0].token_ids) for o in outs) / n, 0),
        wall_s=round(time.time() - t0, 1),
        note="practice accuracy intentionally NOT shown (blind protocol); termination stats only",
        samples=samples,
    )))


if __name__ == "__main__":
    main()
