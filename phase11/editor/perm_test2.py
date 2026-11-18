"""Free: granular permutation test to localize the q anomaly.

Tests (fitness = mean token logprob, baseline ~ -9.77):
  - q permute within head, per layer individually (which layers insensitive?)
  - q permute across heads (swap heads, keep dim order)
  - k permute within head: sliding-only vs full-only
  - q permute: rotated-part only vs unrotated-part only (partial rope)
  - sanity: q row-scaled by random signs (should hurt if q used normally)
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

    def perm_within(w, gen, hd):
        n = w.shape[0] // hd
        w2 = w.view(n, hd, -1)
        idx = torch.randperm(hd, generator=gen)
        return w2[:, idx, :].reshape_as(w)

    def perm_across_heads(w, gen, hd):
        n = w.shape[0] // hd
        w2 = w.view(n, hd, -1)
        idx = torch.randperm(n, generator=gen)
        return w2[idx].reshape_as(w)

    def run(name, fn):
        restore()
        gen = torch.Generator().manual_seed(123)
        fn(gen)
        s = score(model, tok)
        print(f"  {name:44s} {s:8.4f}  delta={s - base:+7.4f}")
        restore()
        return s

    # q within-head, per layer
    print("--- q within-head, single layer at a time ---")
    for li in range(30):
        def fn(gen, li=li):
            hd = 512 if li in FULL else 256
            lm.layers[li].self_attn.q_proj.weight.data.copy_(
                perm_within(orig[li]["q"], gen, hd))
        run(f"q_within L{li:02d}{'(full)' if li in FULL else ''}", fn)

    # k within-head: sliding only vs full only
    print("--- k within-head, group ---")
    def fn_k_slide(gen):
        for li in range(30):
            if li in FULL:
                continue
            lm.layers[li].self_attn.k_proj.weight.data.copy_(
                perm_within(orig[li]["k"], gen, 256))
    run("k_within sliding-only", fn_k_slide)

    def fn_k_full(gen):
        for li in FULL:
            lm.layers[li].self_attn.k_proj.weight.data.copy_(
                perm_within(orig[li]["k"], gen, 512))
    run("k_within full-only", fn_k_full)

    # q across heads (swap whole heads)
    print("--- q across-heads ---")
    def fn_q_across(gen):
        for li in range(30):
            hd = 512 if li in FULL else 256
            lm.layers[li].self_attn.q_proj.weight.data.copy_(
                perm_across_heads(orig[li]["q"], gen, hd))
    run("q_across_heads all layers", fn_q_across)

    # q sign-flip sanity (should hurt if q is used)
    def fn_q_sign(gen):
        for li in range(30):
            w = orig[li]["q"].clone()
            sgn = torch.where(torch.rand(w.shape[0], generator=gen) > 0.5,
                              1.0, -1.0).unsqueeze(1)
            lm.layers[li].self_attn.q_proj.weight.data.copy_(w * sgn)
    run("q_row_signflip (sanity)", fn_q_sign)

    # q row-scale noise (sanity)
    def fn_q_scale(gen):
        for li in range(30):
            w = orig[li]["q"].clone()
            sc = 0.5 + torch.rand(w.shape[0], generator=gen)  # [0.5,1.5]
            lm.layers[li].self_attn.q_proj.weight.data.copy_(w * sc.unsqueeze(1))
    run("q_row_scalen0ise (sanity)", fn_q_scale)

    restore()


if __name__ == "__main__":
    main()
