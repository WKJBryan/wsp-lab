"""Causal layer-skip test (free, GPU, transformers): replace a decoder layer's
output with its input (identity) and measure true next-token ranks.
If skipping layer X restores correct answers, X's block is the destroyer.

Usage (in WSL): ~/wsp3c-venv/bin/python skip_test.py
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
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return ids[0]


# locate decoder layers
layers = None
for name, mod in model.named_modules():
    if name.endswith("language_model.layers"):
        layers = mod
        break
assert layers is not None
print("decoder layers found:", len(layers))

HANDLE = {"h": []}


@torch.no_grad()
def ranks_for(skip):
    for h in HANDLE["h"]:
        h.remove()
    HANDLE["h"] = []

    if skip is not None:
        for s in ([skip] if isinstance(skip, int) else skip):
            mod = layers[s]

            def mk(m):
                def hook(mod, args, out):
                    if isinstance(out, tuple):
                        return (args[0],) + out[1:]
                    return args[0]
                return hook

            HANDLE["h"].append(mod.register_forward_hook(mk(mod)))

    res = []
    for prompt, target in CASES:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
        lg = model(ids).logits[0, -1].float()
        t = tid(target)
        res.append(int((lg > lg[t]).sum()) + 1)
    return res


def main():
    print("skip      | ranks (Paris, Celsius, 4, Jupiter, Au)")
    base = ranks_for(None)
    print(f"{'none':>8} | {base}")
    for s in [24, 25, 26, 27, 28, 29, (26, 27), (27, 28), (28, 29), (26, 27, 28, 29)]:
        r = ranks_for(s)
        print(f"{str(s):>8} | {r}")
    # restore clean state
    ranks_for(None)


if __name__ == "__main__":
    main()
