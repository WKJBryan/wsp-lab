"""Permutation-wound detector: correlation of feature-importance profiles.

For trained nets, per-feature (gate/up row importance) correlates with the
matching down_proj column importance. A permuted/noised down_proj breaks the
alignment while preserving marginals -> correlation collapses.
Compute for dense mlp (30 layers) and each expert (8 x 30).
"""
import torch
from safetensors import safe_open

PATH = "/home/user/wsp11/sandbox/model/model.safetensors"

def prof_importance(g, u):
    # g,u: [F, H] -> per-feature scalar importance
    a = g.float().norm(dim=1)
    b = u.float().norm(dim=1)
    return a * b

def prof_down(d):
    # d: [H, F] -> per-feature (column) importance
    return d.float().norm(dim=0)

def spearman(x, y):
    xr = x.argsort().argsort().float()
    yr = y.argsort().argsort().float()
    xr = (xr - xr.mean()) / (xr.std() + 1e-9)
    yr = (yr - yr.mean()) / (yr.std() + 1e-9)
    return float((xr * yr).mean())

with safe_open(PATH, framework="pt") as sf:
    print("=== dense mlp: spearman(gate*up row-norms, down col-norms) ===")
    for L in range(30):
        gp = sf.get_tensor(f"model.language_model.layers.{L}.mlp.gate_proj.weight")
        up = sf.get_tensor(f"model.language_model.layers.{L}.mlp.up_proj.weight")
        dn = sf.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight")
        a = prof_importance(gp, up)
        b = prof_down(dn)
        # also correlation of squares (variance-weighted)
        print(f"L{L:02d} rho={spearman(a,b):+.3f}")

    print("\n=== experts: per-expert rho ===")
    for L in range(30):
        gu = sf.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj")
        dn = sf.get_tensor(f"model.language_model.layers.{L}.experts.down_proj")
        line = [f"L{L:02d}"]
        for e in range(8):
            g = gu[e, :704]
            u = gu[e, 704:]
            d = dn[e]
            a = prof_importance(g, u)
            b = prof_down(d)
            line.append(f"e{e}:{spearman(a,b):+.2f}")
        print(" ".join(line))

    print("\n=== attention: q-head profile vs o_proj column blocks (16 heads) ===")
    for L in range(30):
        qp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.q_proj.weight")
        op = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.o_proj.weight")
        hd = qp.shape[0] // 16
        a = qp.float().norm(dim=1).view(16, hd).mean(dim=1)
        b = op.float().norm(dim=0).view(16, hd).mean(dim=1)
        print(f"L{L:02d} rho_qo={spearman(a,b):+.3f}")

    print("\n=== attention: k/v profiles (8 kv heads, sliding layers) ===")
    for L in range(30):
        kp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.k_proj.weight")
        vp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.v_proj.weight")
        if vp is None or vp.numel() == 0:
            continue
        nkv = kp.shape[0] // (vp.shape[0] // 8 if vp.shape[0] == kp.shape[0] else 8)
        hd = kp.shape[0] // 8 if kp.shape[0] == 2048 else kp.shape[0] // 2
        nkv = kp.shape[0] // hd
        a = kp.float().norm(dim=1).view(nkv, hd).mean(dim=1)
        b = vp.float().norm(dim=1).view(nkv, hd).mean(dim=1)
        print(f"L{L:02d} nkv={nkv} rho_kv={spearman(a,b):+.3f}")
