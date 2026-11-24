"""Scan small control tensors (scalars, norms, router) for anomalies. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_small.py
"""
import re

import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    ks = list(f.keys())
    print("=== layer_scalar (per layer) ===")
    for k in ks:
        if k.endswith("layer_scalar"):
            t = f.get_tensor(k).float()
            lay = re.search(r"\.(\d+)\.", k).group(1)
            print(f"L{lay:>2}: {t.tolist()}")
    print("\n=== router ===")
    for k in ks:
        if ".router." in k:
            t = f.get_tensor(k).float()
            lay = re.search(r"\.(\d+)\.", k).group(1)
            if k.endswith("per_expert_scale") or k.endswith("router.scale"):
                print(f"L{lay:>2} {k.split('.router.')[1]:>18}: "
                      f"min={t.min():.4g} max={t.max():.4g} mean={t.mean():.4g} "
                      f"vals={ [round(v,4) for v in t.tolist()] if t.numel()<=16 else '' }")
            else:
                print(f"L{lay:>2} {k.split('.router.')[1]:>18}: "
                      f"min={t.min():.4g} max={t.max():.4g} mean={t.mean():.4g} std={t.std():.4g}")
    print("\n=== norm weights (mean/std per group) ===")
    groups = {}
    for k in ks:
        m = re.search(r"layers\.(\d+)\.((?:[\w.]+))\.weight$", k)
        if m and ("norm" in k):
            t = f.get_tensor(k).float()
            g = re.sub(r"\.\d+\.", ".N.", k)
            groups.setdefault(g, []).append((int(m.group(1)), t))
    for g, members in sorted(groups.items()):
        stats = []
        for lay, t in members:
            stats.append((lay, t.mean().item(), t.std().item(), t.min().item(), t.max().item()))
        means = [s[1] for s in stats]
        print(f"{g} x{len(members)}")
        for lay, mn, sd, lo, hi in stats:
            print(f"   L{lay:>2}: mean={mn:.4f} std={sd:.4f} min={lo:.4f} max={hi:.4f}")
    print("\n=== final norm ===")
    t = f.get_tensor("model.language_model.norm.weight").float()
    print(f"mean={t.mean():.4f} std={t.std():.4f} min={t.min():.4f} max={t.max():.4f}")
