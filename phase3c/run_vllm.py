"""Phase 3c: vLLM-based evaluation (the correct engine), run INSIDE WSL2.

Mirrors phase3b protocol: temp 0.6, top-p 0.95, top-k 20, presence_penalty 1.5 (native),
thinking on, 1 sample, cap 32768, boxed-only scoring via phase3b/scorer.py.

Usage (from WSL): python run_vllm.py --model ~/wsp3c/stock --name p3c_stock --n 500
                   python run_vllm.py --model ~/wsp3c/stock --name probe_stock --n 4 --probe
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

CAP = 32768
SEED = 20261105


def extract_last_boxed(text: str):
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--problems", default="problems.json")
    ap.add_argument("--scorer", default="scorer.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.scorer).resolve().parent))
    from scorer import answers_match  # noqa: E402

    problems = json.loads(Path(args.problems).read_text())[: args.n]

    llm = LLM(
        model=args.model,
        max_model_len=40000,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        max_tokens=CAP, seed=SEED,
    )

    msgs = [[{"role": "user", "content": p["q"]}] for p in problems]
    t0 = time.time()
    outs = llm.chat(
        msgs, sp,
        chat_template_kwargs={"enable_thinking": True},
    )

    rows, correct, capped = [], 0, 0
    texts = {}
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        n_tok = len(o.outputs[0].token_ids)
        finish = o.outputs[0].finish_reason
        boxed = extract_last_boxed(text)
        ok = answers_match(p["gold"], boxed)
        correct += ok
        capped += finish == "length"
        rows.append(dict(id=p["id"], ok=ok, finish=finish, boxed=boxed, gold=p["gold"],
                         n_tokens=n_tok, closed="</think>" in text))
        texts[p["id"]] = text
        if args.probe:
            closed = "</think>" in text
            print(f"\n=== {p['id']}: tokens={n_tok} finish={finish} closed={closed} boxed={boxed!r}")
            print("HEAD:", repr(text[:200]))
            if closed:
                i = text.find("</think>")
                print("AROUND-CLOSE:", repr(text[max(0, i - 100): i + 250]))
            else:
                print("TAIL:", repr(text[-250:]))

    n = len(problems)
    summary = dict(name=args.name, model=args.model, engine="vllm", cap=CAP, n=n,
                   accuracy=round(correct / n * 100, 1),
                   cap_hit_rate=round(capped / n * 100, 1),
                   mean_tokens=round(sum(r["n_tokens"] for r in rows) / n, 0),
                   wall_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    out = Path(args.out) if args.out else Path(f"{args.name}.json")
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    Path(str(out).replace(".json", "_texts.json")).write_text(json.dumps(texts, indent=1))


if __name__ == "__main__":
    main()
