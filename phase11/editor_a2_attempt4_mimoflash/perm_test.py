"""Free: permutation-sensitivity test.

Logic: if tensor rows/columns that must ALIGN (gate↔up↔down in SwiGLU,
q↔k per head-dim for RoPE) are INTACT, applying a random permutation to one
partner destroys function → fitness drops a lot. If they are ALREADY
permuted (damage), applying another random permutation keeps them equally
broken → fitness ≈ unchanged.

Fitness = mean token logprob on a fixed set (smooth, sensitive).

Usage: ~/wsp3c-venv/bin/python perm_test.py
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

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


def perm_rows(w, gen):
    idx = torch.randperm(w.shape[0], generator=gen)
    return w[idx]


def perm_cols(w, gen):
    idx = torch.randperm(w.shape[1], generator=gen)
    return w[:, idx]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lm = model.model.language_model

    gen = torch.Generator().manual_seed(0)

    # snapshot
    orig = {}
    for li, l in enumerate(lm.layers):
        orig[li] = {
            "gate": l.mlp.gate_proj.weight.data.clone(),
            "up": l.mlp.up_proj.weight.data.clone(),
            "down": l.mlp.down_proj.weight.data.clone(),
            "gu": l.experts.gate_up_proj.data.clone(),
            "dn": l.experts.down_proj.data.clone(),
            "q": l.self_attn.q_proj.weight.data.clone(),
            "k": l.self_attn.k_proj.weight.data.clone(),
            "o": l.self_attn.o_proj.weight.data.clone(),
        }

    def restore():
        for li, l in enumerate(lm.layers):
            l.mlp.gate_proj.weight.data.copy_(orig[li]["gate"])
            l.mlp.up_proj.weight.data.copy_(orig[li]["up"])
            l.mlp.down_proj.weight.data.copy_(orig[li]["down"])
            l.experts.gate_up_proj.data.copy_(orig[li]["gu"])
            l.experts.down_proj.data.copy_(orig[li]["dn"])
            l.self_attn.q_proj.weight.data.copy_(orig[li]["q"])
            l.self_attn.k_proj.weight.data.copy_(orig[li]["k"])
            l.self_attn.o_proj.weight.data.copy_(orig[li]["o"])

    base = score(model, tok)
    print(f"BASELINE fitness: {base:.4f}")

    # --- static pairing correlations (are gate/up/down aligned in a detectable way?) ---
    print("\n=== static row-norm correlations across neurons (layer 10) ===")
    l = lm.layers[10]
    g = l.mlp.gate_proj.weight.data.float().norm(dim=1)
    u = l.mlp.up_proj.weight.data.float().norm(dim=1)
    d = l.mlp.down_proj.weight.data.float().norm(dim=0)
    print(f"  corr(gate,up)={torch.corrcoef(torch.stack([g, u]))[0,1]:.3f} "
          f"corr(gate,down)={torch.corrcoef(torch.stack([g, d]))[0,1]:.3f} "
          f"corr(up,down)={torch.corrcoef(torch.stack([u, d]))[0,1]:.3f}")
    gu = l.experts.gate_up_proj.data.float()
    # expert 0: first 704 = gate, next 704 = up
    ge = gu[0, :704].norm(dim=1)
    ue = gu[0, 704:].norm(dim=1)
    de = l.experts.down_proj.data.float()[0].norm(dim=0)
    print(f"  expert0 corr(gate,up)={torch.corrcoef(torch.stack([ge, ue]))[0,1]:.3f} "
          f"corr(gate,down)={torch.corrcoef(torch.stack([ge, de]))[0,1]:.3f} "
          f"corr(up,down)={torch.corrcoef(torch.stack([ue, de]))[0,1]:.3f}")

    # --- permutation sensitivity ---
    tests = {}

    def t_mlp_gate():
        for li, l in enumerate(lm.layers):
            l.mlp.gate_proj.weight.data.copy_(
                perm_rows(orig[li]["gate"], gen))
    tests["mlp_gate_rows"] = t_mlp_gate

    def t_mlp_up():
        for li, l in enumerate(lm.layers):
            l.mlp.up_proj.weight.data.copy_(
                perm_rows(orig[li]["up"], gen))
    tests["mlp_up_rows"] = t_mlp_up

    def t_mlp_down():
        for li, l in enumerate(lm.layers):
            l.mlp.down_proj.weight.data.copy_(
                perm_cols(orig[li]["down"], gen))
    tests["mlp_down_cols"] = t_mlp_down

    def t_mlp_gate_only_l10():
        lm.layers[10].mlp.gate_proj.weight.data.copy_(
            perm_rows(orig[10]["gate"], gen))
    tests["mlp_gate_rows_L10only"] = t_mlp_gate_only_l10

    def t_experts_gu():
        for li, l in enumerate(lm.layers):
            gu = orig[li]["gu"].clone()
            # permute intermediate dim within each expert: 1408 = gate704+up704.
            # permute the full 1408 jointly (breaks gate-up split alignment? no—
            # gate/up are chunked from same 1408, so joint permute of 1408 keeps
            # gate-up pairing ONLY if permute applied identically to both halves.
            # We permute gate-half and up-half JOINTLY (same idx) to keep pairing,
            # breaking alignment with down_proj cols.
            idx = torch.randperm(704, generator=gen)
            gate = gu[:, :704][:, idx]
            up = gu[:, 704:][:, idx]
            gu2 = torch.cat([gate, up], dim=1)
            l.experts.gate_up_proj.data.copy_(gu2)
    tests["experts_gu_jointperm_vs_down"] = t_experts_gu

    def t_experts_down():
        for li, l in enumerate(lm.layers):
            dn = orig[li]["dn"].clone()
            idx = torch.randperm(704, generator=gen)
            l.experts.down_proj.data.copy_(dn[:, idx])
    tests["experts_down_cols"] = t_experts_down

    def t_q_within_head():
        # permute q rows WITHIN each head (breaks RoPE dim alignment with k)
        for li, l in enumerate(lm.layers):
            q = orig[li]["q"].clone()
            hd = 512 if li in (5, 11, 17, 23, 29) else 256
            nheads = q.shape[0] // hd
            q2 = q.view(nheads, hd, -1)
            idx = torch.randperm(hd, generator=gen)
            l.self_attn.q_proj.weight.data.copy_(q2[:, idx, :].reshape_as(q))
    tests["q_rows_within_head"] = t_q_within_head

    def t_k_within_head():
        for li, l in enumerate(lm.layers):
            k = orig[li]["k"].clone()
            hd = 512 if li in (5, 11, 17, 23, 29) else 256
            nheads = k.shape[0] // hd
            k2 = k.view(nheads, hd, -1)
            idx = torch.randperm(hd, generator=gen)
            l.self_attn.k_proj.weight.data.copy_(k2[:, idx, :].reshape_as(k))
    tests["k_rows_within_head"] = t_k_within_head

    def t_o_cols_within_head():
        # o_proj input dim = heads*hd; permute columns within each head block
        for li, l in enumerate(lm.layers):
            o = orig[li]["o"].clone()
            hd = 512 if li in (5, 11, 17, 23, 29) else 256
            nheads = o.shape[1] // hd
            o2 = o.view(-1, nheads, hd)
            idx = torch.randperm(hd, generator=gen)
            l.self_attn.o_proj.weight.data.copy_(o2[:, :, idx].reshape_as(o))
    tests["o_cols_within_head"] = t_o_cols_within_head

    for name, fn in tests.items():
        restore()
        gen.manual_seed(0)
        try:
            fn()
            s = score(model, tok)
        except Exception as e:
            print(f"  {name:32s} ERROR {e}")
            restore()
            continue
        delta = s - base
        verdict = "ALREADY-BROKEN?" if abs(delta) < 0.6 else (
            "intact (permutation hurts)" if delta < -0.6 else "improved?!")
        print(f"  {name:32s} fitness={s:8.4f} delta={delta:+7.4f}  -> {verdict}")
    restore()


if __name__ == "__main__":
    main()
