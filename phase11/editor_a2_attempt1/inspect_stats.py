"""Dump safetensors inventory + per-tensor stats for the sandbox model. Free analysis."""
import json

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/stats.json"

rows = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    print(f"total tensors: {len(keys)}")
    meta = f.metadata()
    print("metadata:", meta)
    for k in sorted(keys):
        t = f.get_tensor(k)
        fl = t.float()
        numel = fl.numel()
        absmean = fl.abs().mean().item()
        std = fl.std().item() if numel > 1 else 0.0
        mn = fl.min().item()
        mx = fl.max().item()
        nan = int(torch.isnan(fl).sum().item())
        inf = int(torch.isinf(fl).sum().item())
        zero = int((fl == 0).sum().item())
        rows.append({
            "name": k, "shape": list(t.shape), "dtype": str(t.dtype),
            "numel": numel, "absmean": round(absmean, 6), "std": round(std, 6),
            "min": mn, "max": mx, "nan": nan, "inf": inf,
            "zeros": zero, "zero_frac": round(zero / max(numel, 1), 6),
        })

with open(OUT, "w") as fh:
    json.dump(rows, fh, indent=1)

# print compact per-tensor table sorted by name
for r in rows:
    flag = ""
    if r["nan"] or r["inf"]:
        flag += " **NONFINITE**"
    if r["zero_frac"] > 0.5:
        flag += f" **ZEROS={r['zero_frac']:.3f}**"
    print(f"{r['name']:<70} {str(r['shape']):<22} am={r['absmean']:.5f} std={r['std']:.5f} "
          f"min={r['min']:.4f} max={r['max']:.4f}{flag}")
