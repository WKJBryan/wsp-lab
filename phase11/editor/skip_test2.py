"""Tie-safe causal tests on the PRISTINE model (free, GPU, transformers):
A) skip decoder layers via identity hooks (bypasses block AND scalar).
B) rewrite layer_scalar buffers in-memory (they compound on the residual).
Metric: rank + top-1 token + margin; flags all-tied logits.

Usage (in WSL): ~/wsp3c-venv/bin/python skip_test2.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

CASES = [
    ("The capital of France is", " Paris"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The code is 4711. Question: What is the code? Answer: ", "4"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
]


def tid(s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


layers = None
for name, mod in model.named_modules():
    if name.endswith("language_model.layers"):
        layers = mod
        break

# collect scalar buffers
import re
SCALARS = {int(re.search(r"layers\.(\d+)\.", n).group(1)): b
           for n, b in model.named_buffers() if n.endswith("layer_scalar")}
print("scalar buffers found:", len(SCALARS))
BASE_SC = {k: v.detach().clone() for k, v in SCALARS.items()}
print("base scalars:", [round(float(BASE_SC[i]), 3) for i in range(30)])

HANDLE = {"h": []}


def clear_hooks():
    for h in HANDLE["h"]:
        h.remove()
    HANDLE["h"] = []


def set_skip(spec):
    clear_hooks()
    if spec:
        for s in spec:
            def hook(mod, args, out):
                if isinstance(out, tuple):
                    return (args[0],) + out[1:]
                return args[0]
            HANDLE["h"].append(layers[s].register_forward_hook(hook))


def set_scalars(spec):
    with torch.no_grad():
        for i in range(30):
            v = spec.get(i, float(BASE_SC[i])) if spec else float(BASE_SC[i])
            SCALARS[i].fill_(v)


@torch.no_grad()
def measure():
    res = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        lg = model(ids).logits[0, -1].float()
        t = tid(target)
        tied = bool((lg == lg[0]).all())
        rank = int((lg > lg[t]).sum()) + 1
        top1 = tok.convert_ids_to_tokens([int(lg.argmax())])[0]
        margin = float(lg.max() - lg[t])
        res.append((rank, top1.strip(), round(margin, 2)))
    return res, tied


def show(tag):
    r, tied = measure()
    flag = " [ALL TIED]" if tied else ""
    print(f"{tag:>28}{flag}")
    for (rank, top1, margin), (p, t) in zip(r, CASES):
        print(f"      {t.strip():>9}: rank={rank:>6} top1={top1!r:<12} margin={margin}")


def main():
    print("=== A) layer skips (identity hooks) on pristine scalars ===")
    set_skip(None)
    show("baseline")
    for spec in [(26,), (27,), (28,), (29,), (25, 26), (26, 27), (27, 28),
                 (28, 29), (26, 27, 28), (27, 28, 29)]:
        set_skip(spec)
        show(f"skip {spec}")
    set_skip(None)

    print("\n=== B) scalar-buffer rewrites (no hooks) ===")
    for tag, spec in [
        ("s29=0.82", {29: 0.82}),
        ("s26-29=0.8", {26: 0.8, 27: 0.8, 28: 0.8, 29: 0.8}),
        ("all=0.9", {i: 0.9 for i in range(30)}),
        ("all=0.8", {i: 0.8 for i in range(30)}),
        ("s26-29=0.5", {26: 0.5, 27: 0.5, 28: 0.5, 29: 0.5}),
        ("s24-29=0.8", {i: 0.8 for i in range(24, 30)}),
    ]:
        set_scalars(spec)
        show(tag)
    set_scalars(None)
    show("restored base")


if __name__ == "__main__":
    main()
