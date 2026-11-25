import json, re, sys
from pathlib import Path
import torch
from safetensors import safe_open

MD = Path('/home/user/wsp14/sandbox/model')
files = sorted(MD.glob('*.safetensors'))
tensors = {}
for f in files:
    with safe_open(f, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)

rows = []
for name, t in sorted(tensors.items()):
    if not name.startswith('model.language_model') and not name.startswith('mtp'):
        continue
    tf = t.float()
    rows.append(dict(name=name, shape=list(t.shape), dtype=str(t.dtype),
                     norm=round(float(tf.norm()), 4), mean=round(float(tf.mean()), 6),
                     std=round(float(tf.std()), 6), maxabs=round(float(tf.abs().max()), 6),
                     n_zero=int((tf == 0).sum()), numel=t.numel()))
# print compact table
for r in rows:
    print(f"{r['norm']:>12.4f} {r['std']:>9.5f} {r['maxabs']:>10.5f} {r['n_zero']:>7} {str(r['shape']):<18} {r['name']}")
