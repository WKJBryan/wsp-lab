"""Phase-3b diagnostic pilot: 4 problems x {no penalty, presence penalty}, cap 32768,
full text dump. Decides which component (cap vs penalty) killed the 3b run.

Usage: uv run python phase3b/probe.py
"""

import sys
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_eval3b import PresencePenalty  # noqa: E402

HERE = Path(__file__).resolve().parent
CAP = 32768
N = 4
SEED = 424242


def main():
    ds = load_from_disk(HERE.parent / "phase3" / "data" / "math500")
    problems = [ds[i]["problem"] for i in range(N)]

    tok = AutoTokenizer.from_pretrained("phase3/models/stock")
    model = AutoModelForCausalLM.from_pretrained(
        "phase3/models/stock", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    tok.padding_side = "left"

    for label, penalty in [("NO-PENALTY", None), ("PENALTY-1.5", PresencePenalty(1.5))]:
        torch.manual_seed(SEED)
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
            for q in problems
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=CAP,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                logits_processor=[penalty] if penalty else None,
                pad_token_id=tok.pad_token_id,
            )
        print(f"\n########## {label} ##########")
        for j, q in enumerate(problems):
            new = gen[j][enc["input_ids"].shape[1]:]
            # decode WITH specials so we can see </think> and <think> markers
            raw = tok.decode(new, skip_special_tokens=False)
            closed = "</think>" in raw
            boxed = "\\boxed" in raw
            close_pos = raw.find("</think>")
            print(f"\n--- problem m{j}: tokens={len(new)} closed={closed} "
                  f"close_at_token~{close_pos} boxed={boxed} ({time.time()-t0:.0f}s arm-clock)")
            print("HEAD:", repr(raw[:220]))
            if closed:
                s = max(0, close_pos - 120)
                print("AROUND-CLOSE:", repr(raw[s : close_pos + 260]))
            else:
                print("TAIL:", repr(raw[-260:]))


if __name__ == "__main__":
    main()
