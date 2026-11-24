"""Final-norm gain test (free, GPU, transformers): if g's rogue dims are the
scramble, smoothing g restores answer ranks. Runs on PRISTINE weights.

Usage (in WSL): ~/wsp3c-venv/bin/python g_test2.py
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

G = [p for n, p in model.named_parameters()
     if n.endswith("language_model.norm.weight")][0]
SNAP = G.detach().clone()

CASES = [
    ("The capital of France is", " Paris"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The code is 4711. Question: What is the code? Answer: ", "4"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
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
        rank = int((lg > lg[t]).sum()) + 1
        top1 = tok.convert_ids_to_tokens([int(lg.argmax())])[0].strip()
        res.append((rank, top1, round(float(lg.max() - lg[t]), 2)))
    return res


def show(tag):
    print(f"[{tag}]")
    for (rank, top1, margin), (p, t) in zip(measure(), CASES):
        print(f"    {t.strip():>9}: rank={rank:>6} top1={top1!r:<12} margin={margin}")


@torch.no_grad()
def main():
    gf = SNAP.float()
    med = gf.median()
    print("g stats: mean=%.2f std=%.2f med=%.2f max=%.2f" %
          (gf.mean(), gf.std(), med, gf.max()))
    show("base")
    for tag, new in [
        ("g=mean", torch.full_like(gf, gf.mean())),
        ("g=1", torch.ones_like(gf)),
        ("g=med", torch.full_like(gf, med)),
        ("g clip@3med", gf.clamp(max=3 * med)),
        ("g clip@2med", gf.clamp(max=2 * med)),
        ("g clip@[0.5m,2m]", gf.clamp(min=0.5 * med, max=2 * med)),
    ]:
        G.copy_(new.to(G.dtype))
        show(tag)
    G.copy_(SNAP)
    show("restored")


if __name__ == "__main__":
    main()
