"""Per-matrix row/col norm trend scan across layers (subset-scale fingerprint).
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_trends.py
"""
import re

import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

KINDS = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
         "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
         "experts.gate_up_proj", "experts.down_proj", "router.proj.weight"]


def stats(t):
    t = t.float()
    if t.dim() == 3:
        t = t.reshape(-1, t.shape[-1])
    rn = t.norm(dim=1)
    cn = t.norm(dim=0)
    return (rn.mean().item(), rn.std().item(), rn.max().item(),
            cn.mean().item(), cn.std().item(), cn.max().item())


with safe_open(P, "pt") as f:
    for kind in KINDS:
        rows = []
        for lay in range(30):
            k = f"model.language_model.layers.{lay}.{kind}"
            if k not in f.keys():
                continue
            s = stats(f.get_tensor(k))
            rows.append((lay, s))
        name = kind.split(".")[0] + "." + kind.split(".")[-1]
        print(f"\n=== {kind}")
        print("  L | rowmean rowstd rowmax | colmean colstd colmax")
        for lay, (rm, rs, rmax, cm, cs, cmax) in rows:
            flag = ""
            # z-score vs other layers for each stat
            print(f" {lay:>3} | {rm:7.3f} {rs:6.3f} {rmax:8.2f} | "
                  f"{cm:7.3f} {cs:6.3f} {cmax:8.2f}{flag}")
