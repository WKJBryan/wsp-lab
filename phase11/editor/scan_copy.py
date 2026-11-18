"""Copy/scale detector: compare sorted value multisets of the final-norm gain
against every other same-shape vector. An exact copy (or scaled copy) is a
smoking gun for "copy between tensors" sabotage. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_copy.py
"""
import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    ks = list(f.keys())
    g = f.get_tensor("model.language_model.norm.weight").float()
    gs, _ = g.sort()
    n = g.numel()
    print(f"target: model.norm.weight n={n} mean={g.mean():.4f} std={g.std():.4f}")
    best = []
    for k in ks:
        if k == "model.language_model.norm.weight":
            continue
        t = f.get_tensor(k)
        if t.numel() != n:
            continue
        x = t.float().flatten()
        xs, _ = x.sort()
        diff = (xs - gs).abs().max().item()
        # best scale fit
        c = float((xs * gs).sum() / (xs * xs).sum().clamp(min=1e-12))
        sdiff = (c * xs - gs).abs().max().item()
        best.append((diff, sdiff, c, k))
    best.sort()
    print("\ntop-15 closest by multiset (max|sorted diff|, scaled-diff, scale, name):")
    for diff, sdiff, c, k in best[:15]:
        print(f"  {diff:10.4f} {sdiff:10.4f} c={c:8.4f}  {k}")

    # also: pairwise multiset equality among all 2816-vectors (norms, router.scale)
    vecs = {}
    for k in ks:
        t = f.get_tensor(k)
        if t.numel() == n:
            vecs[k] = t.float().flatten().sort().values
    names = list(vecs)
    print(f"\npairwise multiset scan over {len(names)} vectors (exact matches only):")
    seen = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = (vecs[names[i]] - vecs[names[j]]).abs().max().item()
            if d == 0.0:
                print("  EXACT COPY:", names[i], "<->", names[j])
                seen.add((names[i], names[j]))
    if not seen:
        print("  none")
