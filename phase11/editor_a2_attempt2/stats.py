"""Per-tensor statistics for the sandbox model. Free analysis, not counted.

Usage (inside WSL):
  ~/wsp3c-venv/bin/python /mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/stats.py
"""
import json
import math
from collections import defaultdict

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/stats_out.txt"

rows = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())
    print(f"total tensors: {len(keys)}")
    for k in keys:
        t = f.get_tensor(k)
        tf = t.float()
        n = tf.numel()
        mean = tf.mean().item()
        std = tf.std().item() if n > 1 else 0.0
        mn = tf.min().item()
        mx = tf.max().item()
        nan = torch.isnan(tf).sum().item()
        inf = torch.isinf(tf).sum().item()
        zeros = (tf == 0).sum().item()
        absmean = tf.abs().mean().item()
        absmax = tf.abs().max().item()
        rows.append((k, tuple(t.shape), str(t.dtype), n, mean, std, mn, mx,
                     nan, inf, zeros, absmean, absmax))

with open(OUT, "w") as f:
    f.write(f"{'tensor':<70} {'shape':<28} {'n':>12} {'mean':>10} {'std':>9} "
            f"{'min':>9} {'max':>9} {'nan':>5} {'inf':>5} {'zeros%':>7} {'absmean':>9} {'absmax':>9}\n")
    for (k, shape, dtype, n, mean, std, mn, mx, nan, inf, zeros, absmean, absmax) in rows:
        f.write(f"{k:<70} {str(list(shape)):<28} {n:>12} {mean:>10.4g} {std:>9.3g} "
                f"{mn:>9.3g} {mx:>9.3g} {nan:>5d} {inf:>5d} {100.0*zeros/n:>7.2f} {absmean:>9.3g} {absmax:>9.3g}\n")
print("wrote", OUT)
