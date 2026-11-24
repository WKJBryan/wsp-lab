"""Per-layer Frobenius-norm trend scan of big matrices + embedding row analysis.
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_big.py
"""
import re

import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    ks = list(f.keys())

    def fn(name):
        t = f.get_tensor(name).float()
        return t.norm().item()

    print("=== per-layer matrix F-norms ===")
    print("L  | q_proj  | k_proj  | v_proj  | o_proj  | mlp_dn  | exp_dn  | rout_pr")
    for i in range(30):
        row = [f"{i:>2} "]
        for suf in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                    "mlp.down_proj.weight", "experts.down_proj",
                    "router.proj.weight"]:
            n = f"model.language_model.layers.{i}.{suf}"
            if n in ks:
                row.append(f"{fn(n):8.2f}")
            else:
                row.append("   --   ")
        print(" | ".join(row))

    print("\n=== experts.down_proj per-expert F-norms ===")
    for i in range(30):
        t = f.get_tensor(f"model.language_model.layers.{i}.experts.down_proj").float()
        ns = [round(t[e].norm().item(), 2) for e in range(8)]
        print(f"L{i:>2}: {ns}")

    print("\n=== experts.gate_up_proj per-expert F-norms ===")
    for i in range(30):
        t = f.get_tensor(f"model.language_model.layers.{i}.experts.gate_up_proj").float()
        ns = [round(t[e].norm().item(), 2) for e in range(8)]
        print(f"L{i:>2}: {ns}")

    print("\n=== embed_tokens row norms ===")
    e = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = e.norm(dim=1)
    print(f"row norm: mean={rn.mean():.4f} std={rn.std():.4f} min={rn.min():.4f} "
          f"max={rn.max():.4f} median={rn.median():.4f}")
    # rows far from the median
    med = rn.median()
    dev, idx = (rn - med).abs().sort(descending=True)
    print("top-30 most deviant rows (idx, norm, rowmean, rowstd):")
    for j in idx[:30].tolist():
        r = e[j]
        print(f"  row {j}: norm={rn[j]:.4f} mean={r.mean():.4g} std={r.std():.4g}")
    # NaN/Inf check over everything would be slow; check embedding here
    print("embedding NaN:", torch.isnan(e).any().item(), "Inf:", torch.isinf(e).any().item())
