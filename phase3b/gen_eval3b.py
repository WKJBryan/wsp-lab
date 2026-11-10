"""Phase 3b evaluator: sampled decoding per the original WSP protocol + math-verify scoring.

temp 0.6, top-p 0.95, top-k 20, min_p 0, presence penalty 1.5 (exact custom processor),
thinking on, cap 16384, 1 sample, full MATH-500. Boxed-only scoring; truncated = wrong.

Usage: uv run python phase3b/gen_eval3b.py --model phase3/models/stock --name p3b_stock
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
BATCH = 12  # throughput-only knob (protocol cap unchanged); 32 OOM'd at 16K-token KV cache
CAP = 16384
SEED = 20261105


class PresencePenalty(LogitsProcessor):
    """OpenAI/vLLM-style presence penalty: subtract `penalty` from the logit of
    every token that has appeared at least once (prompt + generated)."""

    def __init__(self, penalty: float):
        self.penalty = penalty
        self.seen = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.seen is None or self.seen.shape[0] != input_ids.shape[0]:
            self.seen = torch.zeros(
                input_ids.shape[0], scores.shape[1], dtype=torch.bool, device=scores.device
            )
            self.seen.scatter_(1, input_ids, True)
        else:
            self.seen.scatter_(1, input_ids[:, -1:].to(torch.int64), True)
        return torch.where(self.seen, scores - self.penalty, scores)


def extract_last_boxed(text: str):
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    from math_verify import parse as mv_parse, verify as mv_verify

    ds = load_from_disk(HERE.parent / "phase3" / "data" / "math500")
    problems = [dict(id=f"m{i}", q=ds[i]["problem"], gold=ds[i]["answer"]) for i in range(500)]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.manual_seed(SEED)
    rows, correct, capped = [], 0, 0
    t0 = time.time()
    for b in range(0, len(problems), BATCH):
        chunk = problems[b : b + BATCH]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p["q"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            for p in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        penalty = PresencePenalty(1.5)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=CAP,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                logits_processor=[penalty],
                pad_token_id=tok.pad_token_id,
            )
        for j, p in enumerate(chunk):
            new_tokens = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            hit_cap = len(new_tokens) >= CAP
            boxed = extract_last_boxed(text)
            ok = False
            if boxed is not None:
                try:
                    ok = bool(mv_verify(mv_parse(f"${p['gold']}$"), mv_parse(f"${boxed}$")))
                except Exception:
                    ok = False
            correct += ok
            capped += hit_cap
            rows.append(dict(id=p["id"], ok=ok, hit_cap=hit_cap, boxed=boxed,
                             gold=p["gold"], n_tokens=len(new_tokens)))
        done = min(b + BATCH, len(problems))
        print(f"  {done}/500 | correct {correct} | cap-hits {capped} | {time.time()-t0:.0f}s", flush=True)

    n = len(problems)
    summary = dict(name=args.name, model=args.model, cap=CAP, n=n,
                   accuracy=round(correct / n * 100, 1), cap_hit_rate=round(capped / n * 100, 1),
                   mean_tokens=round(sum(r["n_tokens"] for r in rows) / n, 0),
                   wall_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    (RUNS / f"{args.name}.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    main()
