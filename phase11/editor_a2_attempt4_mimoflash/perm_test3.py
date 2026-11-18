"""Free: (1) does q matter at all? zero q → uniform scores.
(2) search simple invertible permutations on q / sliding-k that IMPROVE fitness
(assuming damage = simple permutation, its inverse repairs).
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
FULL = {5, 11, 17, 23, 29}

TEXT = [
    ("The capital of France is", " Paris, and it is known for the Eiffel Tower."),
    ("Water boils at", " 100 degrees Celsius at sea level."),
    ("The largest planet in our solar system is", " Jupiter, a gas giant."),
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("A cat is a small", " domesticated feline animal that is often kept as a pet."),
    ("In mathematics, a prime number is", " a natural number greater than 1 that has no positive divisors other than 1 and itself."),
    ("To convert Celsius to Fahrenheit, multiply by", " 9/5 and add 32."),
    ("The sun rises in the", " east and sets in the west."),
    ("The chemical symbol for gold is", " Au."),
]


def score(model, tok):
    total, count = 0.0, 0
    with torch.no_grad():
        for prompt, cont in TEXT:
            p = tok.encode(prompt, add_special_tokens=False)
            c = tok.encode(cont, add_special_tokens=False)
            ids = torch.tensor([p + c]).to(model.device)
            logits = model(ids).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            for i in range(len(p) - 1, len(ids[0]) - 1):
                total += lp[i, ids[0, i + 1]].item()
                count += 1
    return total / count


def apply_perm_rows(w, mode, hd, gen=None):
    """mode: reverse, roll{k}, interleave, sortnorm_asc/desc, rand"""
    n = w.shape[0] // hd
    w2 = w.view(n, hd, -1)
    if mode == "reverse":
        idx = torch.arange(hd - 1, -1, -1)
    elif mode.startswith("roll"):
        k = int(mode[4:])
        idx = torch.roll(torch.arange(hd), -k)
    elif mode == "interleave":
        idx = torch.arange(hd).reshape(-1, 2).t().reshape(-1)
        if hd % 2:
            idx = torch.cat([idx, torch.arange(hd - 1, hd)])
    elif mode == "sortnorm":
        rn = w2.norm(dim=2).mean(dim=0)  # per-dim norm within head (avg over heads)
        idx = torch.argsort(rn)
    elif mode == "sortnorm_desc":
        rn = w2.norm(dim=2).mean(dim=0)
        idx = torch.argsort(rn, descending=True)
    elif mode == "rand":
        idx = torch.randperm(hd, generator=gen)
    else:
        raise ValueError(mode)
    return w2[:, idx, :].reshape_as(w)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lm = model.model.language_model

    orig = {}
    for li, l in enumerate(lm.layers):
        orig[li] = {"q": l.self_attn.q_proj.weight.data.clone(),
                    "k": l.self_attn.k_proj.weight.data.clone()}

    def restore():
        for li, l in enumerate(lm.layers):
            l.self_attn.q_proj.weight.data.copy_(orig[li]["q"])
            l.self_attn.k_proj.weight.data.copy_(orig[li]["k"])

    base = score(model, tok)
    print(f"BASELINE: {base:.4f}")

    # (1) zero q
    restore()
    for li, l in enumerate(lm.layers):
        l.self_attn.q_proj.weight.data.zero_()
    s = score(model, tok)
    print(f"q_zeroed:                    {s:8.4f} delta={s-base:+7.4f}")
    restore()

    # zero k (sliding only) — k also used? sliding k only for scores
    for li, l in enumerate(lm.layers):
        if li not in FULL:
            l.self_attn.k_proj.weight.data.zero_()
    s = score(model, tok)
    print(f"k_zeroed_sliding:            {s:8.4f} delta={s-base:+7.4f}")
    restore()

    # (2) simple permutation candidates on q (all layers)
    print("--- simple perms on q (all layers) ---")
    for mode in ["reverse", "roll1", "roll2", "roll4", "roll8", "roll16",
                 "roll32", "roll64", "roll128", "interleave",
                 "sortnorm", "sortnorm_desc", "rand"]:
        restore()
        gen = torch.Generator().manual_seed(7)
        for li, l in enumerate(lm.layers):
            hd = 512 if li in FULL else 256
            l.self_attn.q_proj.weight.data.copy_(
                apply_perm_rows(orig[li]["q"], mode, hd, gen))
        s = score(model, tok)
        mark = " <== BETTER" if s > base + 0.3 else ""
        print(f"  q_{mode:14s}       {s:8.4f} delta={s-base:+7.4f}{mark}")
    restore()

    # (3) simple permutation candidates on sliding-k
    print("--- simple perms on sliding-k ---")
    for mode in ["reverse", "roll1", "roll2", "roll4", "roll8", "roll16",
                 "roll32", "interleave", "sortnorm", "sortnorm_desc", "rand"]:
        restore()
        gen = torch.Generator().manual_seed(7)
        for li, l in enumerate(lm.layers):
            if li in FULL:
                continue
            l.self_attn.k_proj.weight.data.copy_(
                apply_perm_rows(orig[li]["k"], mode, 256, gen))
        s = score(model, tok)
        mark = " <== BETTER" if s > base + 0.3 else ""
        print(f"  k_{mode:14s}       {s:8.4f} delta={s-base:+7.4f}{mark}")
    restore()

    # (4) also try full-layer k (note: full k doubles as V, so V-order changes too)
    print("--- simple perms on full-k (also V!) ---")
    for mode in ["reverse", "roll8", "interleave", "rand"]:
        restore()
        gen = torch.Generator().manual_seed(7)
        for li in FULL:
            lm.layers[li].self_attn.k_proj.weight.data.copy_(
                apply_perm_rows(orig[li]["k"], mode, 512, gen))
        s = score(model, tok)
        print(f"  kfull_{mode:12s}   {s:8.4f} delta={s-base:+7.4f}")
    restore()


if __name__ == "__main__":
    main()
