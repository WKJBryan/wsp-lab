"""Math-battery scalar sweep (free, GPU, transformers) on pristine weights.
Looks for scalar configs where answer digits enter top-64 (samplable).

Usage (in WSL): ~/wsp3c-venv/bin/python math_scalar_sweep.py
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
    ("17 * 23 =", "391"),
    ("2^10 =", "1024"),
    ("7 * 8 =", "56"),
    ("The positive square root of 144 is", "12"),
    ("100 - 37 =", "63"),
    ("12 * 12 =", "144"),
    ("2^8 =", "256"),
]


def tids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def measure():
    res = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        seq = ids
        ranks = []
        for t in tids(target):
            lg = model(seq).logits[0, -1].float()
            ranks.append(int((lg > lg[t]).sum()) + 1)
            seq = torch.cat([seq, torch.tensor([[t]], device=DEV)], dim=1)
        res.append(ranks)
    return res


def score(res):
    # fraction of digits inside top-64
    allr = [r for rs in res for r in rs]
    return sum(1 for r in allr if r <= 64) / len(allr)


def show(tag):
    res = measure()
    frac = score(res)
    print(f"[{tag}] frac_in_top64={frac:.2f}")
    for (p, t), ranks in zip(CASES, res):
        print(f"    {p[:28]!r:>30} -> {t:>6}: {ranks}")


@torch.no_grad()
def set_scalars(spec):
    for i in range(30):
        SCALARS[i].fill_(spec.get(i, BASE[i]))


def main():
    show("base")
    for tag, spec in [
        ("s26-29=0.8", {i: 0.8 for i in range(26, 30)}),
        ("s26-29=0.95", {i: 0.95 for i in range(26, 30)}),
        ("s20-29=0.95", {i: 0.95 for i in range(20, 30)}),
        ("s26-29=0.95 s0-2=0.5", {**{i: 0.95 for i in range(26, 30)},
                                  **{i: 0.5 for i in range(0, 3)}}),
        ("s24-29=1.0", {i: 1.0 for i in range(24, 30)}),
        ("s26-29=0.8 s0-2=0.5", {**{i: 0.8 for i in range(26, 30)},
                                 **{i: 0.5 for i in range(0, 3)}}),
        ("s26-29=0.8 s0-2=0.3", {**{i: 0.8 for i in range(26, 30)},
                                 **{i: 0.3 for i in range(0, 3)}}),
    ]:
        set_scalars(spec)
        show(tag)
    set_scalars(BASE)
    show("restored")


if __name__ == "__main__":
    main()
