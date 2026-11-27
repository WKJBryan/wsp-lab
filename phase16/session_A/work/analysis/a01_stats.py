"""Per-tensor statistics of the text model, to look for an anomalous tensor. CPU only."""
import json, sys, math
import torch
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
from safetensors import safe_open

path = sys.argv[1]
f = safe_open(path, framework="pt", device="cpu")
keys = [k for k in f.keys() if k.startswith("model.language_model.")]
keys.sort(key=lambda k: (int(k.split(".layers.")[1].split(".")[0]) if ".layers." in k else -1, k))

def kurt(x):
    x = x.float()
    m = x.mean(); s = x.std() + 1e-12
    return float(((x - m) ** 4).mean() / s ** 4 - 3)

print(f"{'tensor':62s} {'shape':>14s} {'mean':>9s} {'std':>8s} {'min':>8s} {'max':>8s} {'kurt':>7s} {'rown_min':>8s} {'rown_max':>8s} {'coln_cv':>7s}")
small = {}
for k in keys:
    t = f.get_tensor(k)
    tf = t.float()
    base = k.replace("model.language_model.", "")
    if t.dim() == 2:
        rn = tf.norm(dim=1); cn = tf.norm(dim=0)
        print(f"{base:62s} {str(tuple(t.shape)):>14s} {tf.mean():9.5f} {tf.std():8.5f} {tf.min():8.3f} {tf.max():8.3f} {kurt(tf):7.2f} {rn.min():8.3f} {rn.max():8.3f} {(cn.std()/cn.mean()).item():7.3f}")
    elif t.dim() == 1:
        small[base] = tf
        print(f"{base:62s} {str(tuple(t.shape)):>14s} {tf.mean():9.5f} {tf.std():8.5f} {tf.min():8.3f} {tf.max():8.3f} {kurt(tf):7.2f}")
    else:
        # conv1d (6144,1,4): per-tap stats
        s = " ".join(f"tap{j}: mean {tf[:, 0, j].mean():+.5f} std {tf[:, 0, j].std():.5f}" for j in range(t.shape[-1]))
        print(f"{base:62s} {str(tuple(t.shape)):>14s} {s}")

print("\n==== small vectors ====")
for k, v in small.items():
    v = v.tolist()
    if len(v) <= 16:
        print(k, [round(x, 4) for x in v])
    else:
        import statistics
        print(k, f"n={len(v)} mean={statistics.mean(v):.4f} min={min(v):.4f} max={max(v):.4f}",
              "first8", [round(x, 3) for x in v[:8]])
