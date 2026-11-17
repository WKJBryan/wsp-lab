"""SVD spectrum spike check (outer-product sabotage) + per-column norm outliers (input-channel scaling)."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"

def top_sv(t, k=4):
    t = t.float()
    if t.ndim == 3:  # experts: flatten expert dim into rows
        t = t.reshape(-1, t.shape[-1])
    m, n = t.shape
    if min(m, n) < k + 1:
        return None
    # gram of smaller dim
    if m >= n:
        g = t.T @ t
    else:
        g = t @ t.T
    try:
        ev = torch.linalg.eigvalsh(g)
    except Exception:
        return None
    sv = ev.flip(0)[:k].clamp(min=0).sqrt()
    return sv

def colnorms(t, dim_in=1):
    return t.float().norm(dim=dim_in)

with safe_open(MODEL, framework="pt", device="cpu") as f:
    print("=== top-4 singular values per matrix (spike if sigma1/sigma2 >> 2) ===")
    flagged = []
    for L in range(30):
        row = []
        for suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                       "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                       "mlp.down_proj.weight", "router.proj.weight"]:
            try:
                t = f.get_tensor(f"model.language_model.layers.{L}.{suffix}")
            except Exception:
                continue
            sv = top_sv(t)
            if sv is None:
                continue
            ratio = (sv[0] / sv[1]).item()
            row.append(f"{suffix.split('.')[-2][:5]}:{sv[0]:.1f}/{sv[1]:.1f}={ratio:.2f}")
            if ratio > 3:
                flagged.append((L, suffix, ratio))
        gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj")
        sv = top_sv(gu.reshape(-1, gu.shape[-1]))
        dn = f.get_tensor(f"model.language_model.layers.{L}.experts.down_proj").transpose(1, 2).reshape(-1, 704)
        sv2 = top_sv(dn)
        print(f"L{L:<2} " + " ".join(row) + f" | expgu:{(sv[0]/sv[1]):.2f} expdn:{(sv2[0]/sv2[1]):.2f}")
    print("\nFLAGGED (sigma1/sigma2 > 3):", flagged if flagged else "none")

    print("\n=== per-column (input-channel) norm outliers ===")
    for L in range(30):
        parts = []
        for suffix, dim in [("self_attn.o_proj.weight", 1), ("mlp.down_proj.weight", 1),
                            ("self_attn.q_proj.weight", 1), ("self_attn.k_proj.weight", 1),
                            ("self_attn.v_proj.weight", 1), ("mlp.gate_proj.weight", 1)]:
            try:
                t = f.get_tensor(f"model.language_model.layers.{L}.{suffix}")
            except Exception:
                continue
            cn = colnorms(t)
            med = cn.median()
            ratio = (cn.max() / med).item()
            lowr = (cn.min() / med).item()
            parts.append(f"{suffix.split('.')[-2][:5]}:hi{ratio:.2f}/lo{lowr:.2f}")
        print(f"L{L:<2} " + " ".join(parts))
