"""Fingerprint analysis: per-expert, per-head distributional stats; k_norm structure.

A reinitialized/noised expert or head shows as a distributional outlier among its
siblings (no reference model needed).
"""
import torch
from safetensors import safe_open

PATH = "/home/user/wsp11/sandbox/model/model.safetensors"

def kurt(t):
    tf = t.float()
    m = tf.mean()
    s = tf.std()
    if s == 0:
        return 0.0
    return float(((tf - m) ** 4).mean() / s**4)

def rowstats(t):
    tf = t.float().reshape(t.shape[0], -1)
    rn = tf.norm(dim=1)
    return float(rn.mean()), float(rn.std()), float(rn.min()), float(rn.max())

with safe_open(PATH, framework="pt") as sf:
    print("=== k_norm / q_norm structure per layer ===")
    for L in range(30):
        qk = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.q_norm.weight").float()
        kn = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.k_norm.weight").float()
        print(f"L{L:02d} q_norm: mean={qk.mean():.4f} std={qk.std():.4f} min={qk.min():.4f} max={qk.max():.4f} | "
              f"k_norm: mean={kn.mean():.4f} std={kn.std():.4f} min={kn.min():.4f} max={kn.max():.4f} uniq~{len(kn.unique())}")

    print("\n=== per-expert stats (8 experts per layer) ===")
    for L in range(30):
        gu = sf.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
        dn = sf.get_tensor(f"model.language_model.layers.{L}.experts.down_proj").float()
        line = [f"L{L:02d}"]
        for e in range(8):
            g = gu[e]
            d = dn[e]
            line.append(f"e{e}:std={g.std():.4f}/k={kurt(g):.1f}/rn=({rowstats(g)[0]:.3f},{rowstats(g)[1]:.3f})")
        print(" ".join(line))
        line2 = [f"L{L:02d} dn"]
        for e in range(8):
            d = dn[e]
            line2.append(f"e{e}:std={d.std():.4f}/k={kurt(d):.1f}/rn=({rowstats(d)[0]:.3f},{rowstats(d)[1]:.3f})")
        print(" ".join(line2))

    print("\n=== per-head stats: q_proj (16 heads) per layer ===")
    for L in range(30):
        qp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.q_proj.weight").float()
        hd = qp.shape[0] // 16
        line = [f"L{L:02d}"]
        for h in range(16):
            hh = qp[h * hd:(h + 1) * hd]
            line.append(f"h{h}:{hh.std():.4f}/{kurt(hh):.0f}")
        print(" ".join(line))

    print("\n=== per-head stats: k_proj + v_proj (8 heads, sliding layers) ===")
    for L in range(30):
        kp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.k_proj.weight").float()
        nkv = 2 if kp.shape[0] == 1024 else 8
        hd = kp.shape[0] // nkv
        line = [f"L{L:02d} k(nkv={nkv})"]
        for h in range(nkv):
            hh = kp[h * hd:(h + 1) * hd]
            line.append(f"h{h}:{hh.std():.4f}/{kurt(hh):.0f}")
        vp = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.v_proj.weight")
        line.append("| v:")
        for h in range(nkv):
            hh = vp.float()[h * hd:(h + 1) * hd]
            line.append(f"h{h}:{hh.std():.4f}/{kurt(hh):.0f}")
        print(" ".join(line))

    print("\n=== dense mlp stats per layer (std/kurt of gate) ===")
    for L in range(30):
        gp = sf.get_tensor(f"model.language_model.layers.{L}.mlp.gate_proj.weight").float()
        dp = sf.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").float()
        print(f"L{L:02d} gate std={gp.std():.4f} k={kurt(gp):.1f} | down std={dp.std():.4f} k={kurt(dp):.1f}")

    print("\n=== o_proj per layer ===")
    for L in range(30):
        op = sf.get_tensor(f"model.language_model.layers.{L}.self_attn.o_proj.weight").float()
        print(f"L{L:02d} o std={op.std():.4f} k={kurt(op):.1f} rn=({rowstats(op)[0]:.2f},{rowstats(op)[1]:.2f})")
