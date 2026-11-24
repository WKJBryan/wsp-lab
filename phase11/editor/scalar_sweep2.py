"""Focused scalar sweep v2 + generation check for winners (free, GPU).

Usage (in WSL): ~/wsp3c-venv/bin/python scalar_sweep2.py
"""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

SCALARS = {int(re.search(r"layers\.(\d+)\.", n).group(1)): b
           for n, b in model.named_buffers() if n.endswith("layer_scalar")}
BASE = {k: float(v) for k, v in SCALARS.items()}

CASES = [
    ("The capital of France is", " Paris"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The code is 4711. Question: What is the code? Answer: ", "4"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("Shakespeare wrote Hamlet and", " Macbeth"),
]


def tid(s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


@torch.no_grad()
def measure():
    res = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        lg = model(ids).logits[0, -1].float()
        t = tid(target)
        res.append((int((lg > lg[t]).sum()) + 1,
                    tok.convert_ids_to_tokens([int(lg.argmax())])[0].strip()))
    return res


def gen_text():
    q = "What is 17 * 23? Show your work briefly."
    enc = tok.apply_chat_template([[{"role": "user", "content": q}]],
                                  add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    enc = enc.to(DEV)
    o = model.generate(enc, max_new_tokens=120, do_sample=False)
    return tok.decode(o[0][enc.shape[1]:])[:200]


def show(tag, with_gen=False):
    print(f"[{tag}]")
    for (rank, top1), (p, t) in zip(measure(), CASES):
        print(f"    {t.strip():>9}: rank={rank:>6} top1={top1!r}")
    if with_gen:
        print(f"    gen: {gen_text()!r}")


@torch.no_grad()
def set_scalars(spec):
    for i in range(30):
        SCALARS[i].fill_(spec.get(i, BASE[i]))


def main():
    show("base")
    for tag, spec, wg in [
        ("s26-29=0.98", {i: 0.98 for i in range(26, 30)}, False),
        ("s26-29=1.0", {i: 1.0 for i in range(26, 30)}, False),
        ("s24-29=1.0", {i: 1.0 for i in range(24, 30)}, False),
        ("s26-29=1.0 s0-2=0.5", {**{i: 1.0 for i in range(26, 30)},
                                 **{i: 0.5 for i in range(0, 3)}}, True),
        ("s26-29=0.95 s0-2=0.5", {**{i: 0.95 for i in range(26, 30)},
                                  **{i: 0.5 for i in range(0, 3)}}, False),
        ("s26-29=0.95 s0-2=0.7", {**{i: 0.95 for i in range(26, 30)},
                                  **{i: 0.7 for i in range(0, 3)}}, False),
        ("s20-29=0.95", {i: 0.95 for i in range(20, 30)}, False),
        ("s20-29=0.95 s0-2=0.5", {**{i: 0.95 for i in range(20, 30)},
                                  **{i: 0.5 for i in range(0, 3)}}, True),
    ]:
        set_scalars(spec)
        show(tag, wg)
    set_scalars(BASE)
    show("restored")


if __name__ == "__main__":
    main()
