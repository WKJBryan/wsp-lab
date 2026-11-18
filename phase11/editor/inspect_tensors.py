"""CPU-only tensor inventory of the sandbox checkpoint (free analysis, no GPU).

Usage (in WSL): ~/wsp3c-venv/bin/python inspect_tensors.py [pattern]
Prints name/shape/dtype plus basic stats for matching tensors.
"""
import re
import sys

import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"
pat = sys.argv[1] if len(sys.argv) > 1 else None

with safe_open(P, "pt") as f:
    ks = list(f.keys())
    print("n_tensors", len(ks))
    if pat is None:
        # group by suffix pattern
        groups = {}
        for k in ks:
            g = re.sub(r"\.\d+\.", ".N.", k)
            groups.setdefault(g, []).append(k)
        for g, members in sorted(groups.items()):
            ex = f.get_tensor(members[0])
            print(f"{g}  x{len(members)}  shape={tuple(ex.shape)} dtype={ex.dtype}")
    else:
        rx = re.compile(pat)
        for k in ks:
            if rx.search(k):
                t0 = f.get_tensor(k)
                t = t0.float()
                print(f"{k} shape={tuple(t.shape)} dtype={t0.dtype} "
                      f"mean={t.mean():.4g} std={t.std():.4g} "
                      f"absmax={t.abs().max():.4g} norm={t.norm():.4g}")
