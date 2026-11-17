"""Per-tensor stats for the sandbox model. CPU, sequential, one tensor at a time."""
import json
import struct

import torch
from safetensors import safe_open

PATH = "/home/user/wsp11/sandbox/model/model.safetensors"

with open(PATH, "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))
header.pop("__metadata__", None)

def natkey(name):
    # natural sort: layers.N in numeric order
    parts = name.split(".")
    return [int(p) if p.isdigit() else p for p in parts]

rows = []
with safe_open(PATH, framework="pt") as sf:
    for name in sorted(header.keys(), key=natkey):
        t = sf.get_tensor(name)
        tf = t.float()
        n = tf.numel()
        nzero = int((tf == 0).sum())
        nan = int(torch.isnan(tf).sum())
        inf = int(torch.isinf(tf).sum())
        rows.append({
            "name": name,
            "shape": list(t.shape),
            "mean": float(tf.mean()),
            "std": float(tf.std()) if n > 1 else 0.0,
            "min": float(tf.min()),
            "max": float(tf.max()),
            "absmax": float(tf.abs().max()),
            "norm": float(tf.norm()),
            "pzero": nzero / n,
            "nan": nan,
            "inf": inf,
        })
        del t, tf

for r in rows:
    print(json.dumps(r))
