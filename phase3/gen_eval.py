"""Phase 3 evaluator: batched greedy generation + boxed-answer accuracy + cap-hit rate.

Arms: --arm practice (128 fixed GSM8K-train problems) | math500 | math250 (first 250)
Usage: uv run python phase3/gen_eval.py --model phase3/models/stock --arm practice [--cap 4096]
Writes results json + generations to phase3/runs/<name>.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)

PRACTICE_IDS_SEED = 31337
PRACTICE_N = 64   # addendum 2: passes use the first 64 of the fixed 128 practice ids
BATCH = 32


def norm_answer(s: str) -> str:
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "").replace("\\$", "")
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2]
    return s


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    # fallback: last number in the generation (GSM8K-style answers; equal for all arms)
    nums = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?%?", text)
    if nums:
        return nums[-1].replace("$", "").replace(",", "").replace("%", "")
    return None


def load_problems(arm: str):
    if arm == "practice":
        ds = load_from_disk(HERE / "data" / "gsm8k_train")
        rng = random.Random(PRACTICE_IDS_SEED)
        idx = rng.sample(range(len(ds)), PRACTICE_N)
        return [
            dict(id=f"gsm{i}", q=ds[i]["question"], gold=ds[i]["answer"].split("####")[-1].strip())
            for i in idx
        ]
    ds = load_from_disk(HERE / "data" / "math500")
    n = 500 if arm == "math500" else 250
    return [dict(id=f"m{i}", q=ds[i]["problem"], gold=ds[i]["answer"]) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="practice", choices=["practice", "math500", "math250"])
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cap = args.cap or 6144  # addendum 2: uniform caps
    name = args.name or f"{Path(args.model).name}_{args.arm}_cap{cap}"

    problems = load_problems(args.arm)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out_rows = []
    correct = 0
    capped = 0
    t0 = time.time()
    for b in range(0, len(problems), BATCH):
        chunk = problems[b : b + BATCH]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p["q"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # harness addendum (ISSUES #6): template default disables thinking
            )
            for p in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=cap,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j, p in enumerate(chunk):
            new_tokens = gen[j][enc["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            hit_cap = len(new_tokens) >= cap
            pred = extract_boxed(text)
            ok = pred is not None and norm_answer(pred) == norm_answer(p["gold"])
            correct += ok
            capped += hit_cap
            out_rows.append(dict(id=p["id"], ok=ok, hit_cap=hit_cap, pred=pred, gold=p["gold"],
                                 n_tokens=len(new_tokens), text=text))
        done = min(b + BATCH, len(problems))
        print(f"  {done}/{len(problems)} | acc so far {correct}/{done} | cap-hits {capped}", flush=True)

    n = len(problems)
    summary = dict(name=name, model=str(args.model), arm=args.arm, cap=cap, n=n,
                   accuracy=round(correct / n * 100, 1), cap_hit_rate=round(capped / n * 100, 1),
                   wall_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    (RUNS / f"{name}.json").write_text(json.dumps(dict(summary=summary, rows=out_rows), indent=1))


if __name__ == "__main__":
    main()
