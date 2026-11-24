"""Early-exit sweep (free, GPU, transformers): identity-skip the last N layers
(so the readout sees mid-layer state) and measure MATH-flavored single-token
targets + fact battery + greedy generation. If answers become top-ranked,
the repair = zero the late blocks + set their scalars to 1.0.

Usage (in WSL): ~/wsp3c-venv/bin/python early_exit_test.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()
DEV = "cuda:0"

layers = None
for name, mod in model.named_modules():
    if name.endswith("language_model.layers"):
        layers = mod
        break

CASES = [
    # math-flavored single-token answers
    ("17 * 23 =", "391"),
    ("2^10 =", "1024"),
    ("7 * 8 =", "56"),
    ("The positive square root of 144 is", "12"),
    ("100 - 37 =", "63"),
    # fact battery
    ("The capital of France is", " Paris"),
    ("Water boils at 100 degrees", " Celsius"),
    ("The largest planet in our solar system is", " Jupiter"),
]


def tids(s):
    return tok(s, add_special_tokens=False)["input_ids"]


HANDLE = []


def set_skip(spec):
    global HANDLE
    for h in HANDLE:
        h.remove()
    HANDLE = []
    if spec:
        for s in spec:
            def hook(mod, args, out):
                if isinstance(out, tuple):
                    return (args[0],) + out[1:]
                return args[0]
            HANDLE.append(layers[s].register_forward_hook(hook))


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
        res.append((ranks[0], ranks,
                    tok.convert_ids_to_tokens([int(model(ids).logits[0, -1].argmax())])[0].strip()))
    return res


@torch.no_grad()
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
    return tok.decode(o[0][enc.shape[1]:])[:230]


def show(tag, wg=False):
    print(f"[{tag}]")
    for (rank0, ranks, top1), (p, t) in zip(measure(), CASES):
        print(f"    {p[:32]!r:>34} -> {t.strip():>8}: ranks={ranks} top1={top1!r}")
    if wg:
        print(f"    gen: {gen_text()!r}")


def main():
    show("base")
    for spec in [(29,), (28, 29), (27, 28, 29), (26, 27, 28, 29),
                 (25, 26, 27, 28, 29), (24, 25, 26, 27, 28, 29),
                 (20, 21, 22, 23, 24, 25, 26, 27, 28, 29)]:
        set_skip(spec)
        show(f"skip {spec[0]}..{spec[-1]}", wg=(spec == (25, 26, 27, 28, 29)))
    set_skip(None)
    show("restored")


if __name__ == "__main__":
    main()
