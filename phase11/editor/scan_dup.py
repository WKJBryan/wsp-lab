"""Head-block duplicate detector v2 (shape-aware). CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_dup.py
"""
import torch
import torch.nn.functional as F
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    ks = list(f.keys())

    print("=== per-layer shapes of q/k/v/o ===")
    for lay in [0, 5, 11, 17, 23, 29]:
        shapes = []
        for n in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            k = f"model.language_model.layers.{lay}.self_attn.{n}.weight"
            if k in ks:
                shapes.append(f"{n}{tuple(f.get_tensor(k).shape)}")
            else:
                shapes.append(f"{n}=None")
        print(f"  L{lay:>2}: {' '.join(shapes)}")

    def head_blocks(k, head_dim, axis):
        if k not in ks:
            return []
        t = f.get_tensor(k).float()
        n = t.shape[0] if axis == "rows" else t.shape[1]
        nh = n // head_dim
        out = []
        for i in range(nh):
            if axis == "rows":
                out.append(t[i * head_dim:(i + 1) * head_dim].reshape(-1))
            else:
                out.append(t[:, i * head_dim:(i + 1) * head_dim].reshape(-1))
        return out

    FULL = {5, 11, 17, 23, 29}
    print("\n=== within-layer max pairwise head-block cosine ===")
    for kind, axis in [("self_attn.v_proj.weight", "rows"),
                       ("self_attn.k_proj.weight", "rows"),
                       ("self_attn.q_proj.weight", "rows"),
                       ("self_attn.o_proj.weight", "cols")]:
        worst = (0, None, None, None)
        for lay in range(30):
            hd = 512 if lay in FULL else 256
            if "o_proj" in kind:
                hd = 512 if lay in FULL else 256
            bs = head_blocks(f"model.language_model.layers.{lay}.{kind}", hd, axis)
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    if bs[i].numel() != bs[j].numel():
                        continue
                    c = float(F.cosine_similarity(bs[i], bs[j], dim=0))
                    if c > worst[0]:
                        worst = (c, lay, i, j)
        print(f"  {kind:<28} max cos={worst[0]:.4f} at L{worst[1]} heads {worst[2]}-{worst[3]}")

    print("\n=== cross-layer same-head v_proj block cosine ===")
    vblocks = {}
    for lay in range(30):
        bs = head_blocks(f"model.language_model.layers.{lay}.self_attn.v_proj.weight", 256, "rows")
        if bs:
            vblocks[lay] = bs
    worst = (0, None, None, None)
    lays = sorted(vblocks)
    for a in range(len(lays)):
        for b in range(a + 1, len(lays)):
            for h in range(8):
                c = float(F.cosine_similarity(vblocks[lays[a]][h], vblocks[lays[b]][h], dim=0))
                if c > worst[0]:
                    worst = (c, lays[a], lays[b], h)
    print(f"  max cos={worst[0]:.4f} (L{worst[1]}-L{worst[2]} head {worst[3]})")

    print("\n=== within-layer k-vs-v block cosine ===")
    worst = (0, None, None)
    for lay in range(30):
        kb = head_blocks(f"model.language_model.layers.{lay}.self_attn.k_proj.weight", 256, "rows")
        vb = head_blocks(f"model.language_model.layers.{lay}.self_attn.v_proj.weight", 256, "rows")
        for h in range(min(len(kb), len(vb))):
            c = float(F.cosine_similarity(kb[h], vb[h], dim=0))
            if c > worst[0]:
                worst = (c, lay, h)
    print(f"  max cos={worst[0]:.4f} at L{worst[1]} head {worst[2]}")
