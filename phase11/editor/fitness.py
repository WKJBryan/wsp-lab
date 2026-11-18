"""Free: smooth fitness metric = mean per-token logprob over a fixed set of
high-confidence continuations (raw text + arithmetic + chat QA).
Lower (less negative) is better. One model load, many patches per run.

Usage: ~/wsp3c-venv/bin/python fitness.py [patch ...]
Patches are applied on top of each other and reverted between trials.
"""

import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

# (prompt, continuation) — continuations a healthy model should find easy
TEXT = [
    ("The capital of France is", " Paris, and it is known for the Eiffel Tower."),
    ("Water boils at", " 100 degrees Celsius at sea level."),
    ("The largest planet in our solar system is", " Jupiter, a gas giant."),
    ("The sun rises in the", " east and sets in the west."),
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("The chemical symbol for gold is", " Au."),
    ("There are seven days in a", " week, from Monday to Sunday."),
    ("A cat is a small", " domesticated feline animal that is often kept as a pet."),
    ("The meaning of life is", " a philosophical question that has been debated for centuries."),
    ("In mathematics, a prime number is", " a natural number greater than 1 that has no positive divisors other than 1 and itself."),
    ("To convert Celsius to Fahrenheit, multiply by", " 9/5 and add 32."),
]

CHAT_QA = [
    ("What is the capital of France?", " Paris"),
    ("What is 17 * 23?", " 391"),
    ("How many days are in a week?", " 7"),
    ("What planet is largest in our solar system?", " Jupiter"),
]


def make_patches():
    def p_final_scale(c):
        def f(m):
            m.model.language_model.norm.weight.data.mul_(c)
        return f

    def p_final_median():
        def f(m):
            w = m.model.language_model.norm.weight.data
            w.fill_(w.float().median().to(w.dtype))
        return f

    def p_final_clip(c):
        def f(m):
            m.model.language_model.norm.weight.data.clamp_(max=c)
        return f

    def p_family(name, c):
        def f(m):
            for l in m.model.language_model.layers:
                mod = getattr(l, name, None)
                if mod is not None and hasattr(mod, "weight"):
                    mod.weight.data.mul_(c)
        return f

    def p_family_ones(name):
        def f(m):
            for l in m.model.language_model.layers:
                mod = getattr(l, name, None)
                if mod is not None and hasattr(mod, "weight"):
                    mod.weight.data.fill_(1.0)
        return f

    def p_layer_scalar_scale(c):
        def f(m):
            for l in m.model.language_model.layers:
                l.layer_scalar.mul_(c)
        return f

    def p_embed_scale(c):
        def f(m):
            m.model.language_model.embed_tokens.weight.data.mul_(c)
        return f

    def p_knorm_scale(c):
        def f(m):
            for l in m.model.language_model.layers:
                l.self_attn.k_norm.weight.data.mul_(c)
        return f

    def p_qnorm_scale(c):
        def f(m):
            for l in m.model.language_model.layers:
                l.self_attn.q_norm.weight.data.mul_(c)
        return f

    def p_l0_postff_scale(c):
        def f(m):
            l0 = m.model.language_model.layers[0]
            l0.post_feedforward_layernorm.weight.data.mul_(c)
        return f

    def p_l0_postattn_scale(c):
        def f(m):
            l0 = m.model.language_model.layers[0]
            l0.post_attention_layernorm.weight.data.mul_(c)
        return f

    P = {}
    for c in (0.25, 0.5, 0.75, 1.5, 2.0, 4.0):
        P[f"final_x{c}"] = p_final_scale(c)
        P[f"inputln_x{c}"] = p_family("input_layernorm", c)
        P[f"preff_x{c}"] = p_family("pre_feedforward_layernorm", c)
        P[f"preff2_x{c}"] = p_family("pre_feedforward_layernorm_2", c)
        P[f"postff1_x{c}"] = p_family("post_feedforward_layernorm_1", c)
        P[f"postff2_x{c}"] = p_family("post_feedforward_layernorm_2", c)
        P[f"postattn_x{c}"] = p_family("post_attention_layernorm", c)
        P[f"postff_x{c}"] = p_family("post_feedforward_layernorm", c)
        P[f"lscalar_x{c}"] = p_layer_scalar_scale(c)
        P[f"embed_x{c}"] = p_embed_scale(c)
        P[f"knorm_x{c}"] = p_knorm_scale(c)
        P[f"qnorm_x{c}"] = p_qnorm_scale(c)
        P[f"l0postff_x{c}"] = p_l0_postff_scale(c)
        P[f"l0postattn_x{c}"] = p_l0_postattn_scale(c)
    P["final_median"] = p_final_median()
    for name in ("input_layernorm", "pre_feedforward_layernorm",
                 "pre_feedforward_layernorm_2", "post_attention_layernorm",
                 "post_feedforward_layernorm"):
        P[f"{name}_ones"] = p_family_ones(name)
    return P


def score(model, tok):
    total = 0.0
    count = 0
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
        for q, a in CHAT_QA:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            ids = {k: v.to(model.device) for k, v in ids.items()}
            pref = ids["input_ids"].shape[1]
            c = tok.encode(a, add_special_tokens=False)
            full = torch.cat([ids["input_ids"], torch.tensor([c]).to(model.device)], dim=1)
            logits = model(full).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            for i in range(pref - 1, len(full[0]) - 1):
                total += lp[i, full[0, i + 1]].item()
                count += 1
    return total / count, total


def main():
    patches = make_patches()
    names = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # snapshot all tensors we might touch
    orig = {}
    lm = model.model.language_model
    orig["final"] = lm.norm.weight.data.clone()
    orig["embed"] = lm.embed_tokens.weight.data.clone()
    orig["lscalar"] = torch.stack([l.layer_scalar.clone() for l in lm.layers])
    for name in ("input_layernorm", "pre_feedforward_layernorm",
                 "pre_feedforward_layernorm_2", "post_attention_layernorm",
                 "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                 "post_feedforward_layernorm_2"):
        orig[name] = [getattr(l, name).weight.data.clone() for l in lm.layers]
    orig["k"] = [l.self_attn.k_norm.weight.data.clone() for l in lm.layers]
    orig["q"] = [l.self_attn.q_norm.weight.data.clone() for l in lm.layers]

    def restore():
        lm.norm.weight.data.copy_(orig["final"])
        lm.embed_tokens.weight.data.copy_(orig["embed"])
        for i, l in enumerate(lm.layers):
            l.layer_scalar.copy_(orig["lscalar"][i])
            for name in orig:
                if name in ("final", "embed", "lscalar", "k", "q"):
                    continue
                getattr(l, name).weight.data.copy_(orig[name][i])
            l.self_attn.k_norm.weight.data.copy_(orig["k"][i])
            l.self_attn.q_norm.weight.data.copy_(orig["q"][i])

    m0, t0 = score(model, tok)
    print(f"BASELINE: mean_lp={m0:.4f} total={t0:.1f}")

    if not names:
        # rank all patches
        results = []
        for name, fn in patches.items():
            restore()
            fn(model)
            m, t = score(model, tok)
            results.append((m, name))
            print(f"  {name:28s} mean_lp={m:.4f} delta={m - m0:+.4f}")
        restore()
        print("\nBEST:")
        for m, name in sorted(results, reverse=True)[:8]:
            print(f"  {name:28s} {m:.4f} ({m - m0:+.4f})")
        return

    for name in names:
        if name not in patches:
            print("unknown:", name)
            continue
        restore()
        patches[name](model)
        m, t = score(model, tok)
        print(f"  {name:28s} mean_lp={m:.4f} delta={m - m0:+.4f}")
    restore()


if __name__ == "__main__":
    main()
