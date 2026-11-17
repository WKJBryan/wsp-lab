"""Edit a tensor in a COPY of the sandbox model under /home/user/wsp11/sandbox/.

Usage:
  ~/wsp3c-venv/bin/python edit_model.py <src_dir> <dst_dir> <edit_spec> [<edit_spec> ...]

edit_spec grammar (applied in order):
  scale:<tensor_substr>:<factor>          multiply matching tensor(s) by factor
  rowscale:<tensor_substr>:<factor>       multiply TINY rows (<median/6) by factor
  rowfill:<tensor_substr>:<mult>          rescale tiny rows to median*mult
  zero:<tensor_substr>                    set matching tensor(s) to zero
  layerscalar:<idx>:<value>               set layer_scalar of layer idx
Copies the source dir first (hardlink safetensors where possible then rewrite edited shard).
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
specs = sys.argv[3:]
assert src != dst
os.makedirs(dst, exist_ok=True)

# copy non-weight files; model.safetensors will be rewritten fully after edit
for fn in os.listdir(src):
    if fn == "model.safetensors":
        continue
    s = os.path.join(src, fn)
    d = os.path.join(dst, fn)
    if not os.path.exists(d):
        shutil.copy2(s, d)

tensors = load_file(os.path.join(src, "model.safetensors"))
log = []

def matches(k, sub):
    return sub in k and ("language_model" in k or "embed" in k)

for spec in specs:
    parts = spec.split(":")
    op = parts[0]
    if op == "scale":
        sub, factor = parts[1], float(parts[2])
        for k in list(tensors):
            if matches(k, sub):
                tensors[k] = (tensors[k].float() * factor).to(tensors[k].dtype)
                log.append(f"{op} {k} x{factor}")
    elif op == "rowscale":
        sub, factor = parts[1], float(parts[2])
        for k in list(tensors):
            if matches(k, sub):
                t = tensors[k].float()
                if t.dim() == 2 and t.shape[0] >= 8:
                    rn = t.pow(2).sum(1).sqrt()
                    med = rn.median()
                    mask = rn < med / 6
                    t[mask] *= factor
                    tensors[k] = t.to(tensors[k].dtype)
                    log.append(f"{op} {k} tiny_rows x{factor} (n={int(mask.sum())})")
    elif op == "rowfill":
        sub, mult = parts[1], float(parts[2])
        for k in list(tensors):
            if matches(k, sub):
                t = tensors[k].float()
                if t.dim() == 2 and t.shape[0] >= 8:
                    rn = t.pow(2).sum(1).sqrt()
                    med = rn.median()
                    mask = rn < med / 6
                    scale = (med * mult) / rn[mask]
                    t[mask] *= scale.unsqueeze(1)
                    tensors[k] = t.to(tensors[k].dtype)
                    log.append(f"{op} {k} tiny_rows -> med*{mult} (n={int(mask.sum())})")
    elif op == "zero":
        sub = parts[1]
        for k in list(tensors):
            if matches(k, sub):
                tensors[k] = torch.zeros_like(tensors[k])
                log.append(f"{op} {k}")
    elif op == "embscale":
        # embscale:<comma ids>:<factor>
        ids = [int(x) for x in parts[1].split(",")]
        factor = float(parts[2])
        k = "model.language_model.embed_tokens.weight"
        t = tensors[k].float()
        for i in ids:
            t[i] *= factor
        tensors[k] = t.to(tensors[k].dtype)
        log.append(f"{op} rows {ids[:12]}{'...' if len(ids) > 12 else ''} x{factor}")
    elif op == "layerscalar":
        idx, val = int(parts[1]), float(parts[2])
        k = f"model.language_model.layers.{idx}.layer_scalar"
        tensors[k] = torch.tensor([val], dtype=tensors[k].dtype)
        log.append(f"{op} {k} = {val}")
    else:
        raise SystemExit(f"unknown op {op}")

save_file(tensors, os.path.join(dst, "model.safetensors"),
          metadata={"format": "pt"})
print(json.dumps({"dst": dst, "edits": log}, indent=1))
