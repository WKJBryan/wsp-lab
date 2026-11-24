"""Rogue-dim alignment detector for norm-gain vectors (element-permutation check).
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_align.py
"""
import re

import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    ks = [k for k in f.keys() if re.search(r"layers\.\d+\.", k) and k.endswith(".weight")
          and f.get_tensor(k).dim() == 1 and f.get_tensor(k).numel() == 2816]
    groups = {}
    for k in ks:
        g = re.sub(r"layers\.\d+\.", "layers.N.", k)
        groups.setdefault(g, []).append(k)

    for g, members in sorted(groups.items()):
        tops = {}
        for k in members:
            t = f.get_tensor(k).float()
            lay = int(re.search(r"layers\.(\d+)\.", k).group(1))
            top = set(torch.topk(t, 10).indices.tolist())
            tops[lay] = top
        lays = sorted(tops)
        union = set()
        inter = set(tops[lays[0]])
        for l in lays:
            union |= tops[l]
            inter &= tops[l]
        # overlap of each layer with layer0
        ovs = [len(tops[l] & tops[lays[0]]) for l in lays]
        print(f"{g}: union={len(union)} inter={len(inter)} "
              f"overlap_with_L{lays[0]}: {ovs}")
        if len(union) <= 30:
            cnt = {}
            for l in lays:
                for d in tops[l]:
                    cnt[d] = cnt.get(d, 0) + 1
            common = sorted(cnt.items(), key=lambda x: -x[1])[:12]
            print(f"    frequent top-dims (dim, in N layers): {common}")

    print("\nfinal norm top-10 dims:")
    t = f.get_tensor("model.language_model.norm.weight").float()
    print("   ", torch.topk(t, 10).indices.tolist())
    print("    values:", [round(v, 1) for v in torch.topk(t, 10).values.tolist()])
    print("\nrouter.scale top-10 dims (layer 0):")
    t = f.get_tensor("model.language_model.layers.0.router.scale").float()
    print("   ", torch.topk(t, 10).indices.tolist())
