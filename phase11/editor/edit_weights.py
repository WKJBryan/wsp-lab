"""Safe weight-edit tool for the sandbox checkpoint.

Always restores model.safetensors from model.safetensors.bak first (unless
--no-restore), then applies the ops in the given recipe JSON. This makes every
experiment reproducible and reversible.

Usage (in WSL):
  ~/wsp3c-venv/bin/python edit_weights.py recipe.json [--no-restore]

Recipe = JSON list of ops:
  {"op": "scale_tensor",  "name": N, "factor": f}
  {"op": "fill_tensor",   "name": N, "value": v}          # broadcast fill (vectors)
  {"op": "add_const",     "name": N, "value": v}
  {"op": "scale_rows",    "name": N, "rows": [..], "factor": f}   # dim0 rows
  {"op": "scale_cols",    "name": N, "cols": [..], "factor": f}   # dim1 cols
  {"op": "zero_rows",     "name": N, "rows": [..]}
  {"op": "set_index",     "name": N, "index": [..], "value": v}
  {"op": "copy_from",     "name": N, "src": M}            # copy whole tensor
  {"op": "row_norms_report", "name": N}                    # no-op, just report
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = Path("/home/user/wsp11/sandbox/model")
SHARD = MODEL_DIR / "model.safetensors"
BAK = MODEL_DIR / "model.safetensors.bak"


def main():
    recipe_path = sys.argv[1]
    restore = "--no-restore" not in sys.argv
    ops = json.loads(Path(recipe_path).read_text())

    if restore:
        if not BAK.exists():
            print("ERROR: backup missing:", BAK)
            sys.exit(1)
        print("restoring from backup ...")
        shutil.copyfile(BAK, SHARD)

    with safe_open(str(SHARD) if not restore else str(BAK), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    for op in ops:
        kind = op["op"]
        if kind in ("fill_match", "scale_match"):
            import re as _re
            rx = _re.compile(op["pattern"])
            hits = [k for k in tensors if rx.search(k)]
            for name in hits:
                t = tensors[name]
                if kind == "fill_match":
                    tensors[name] = torch.full_like(t, op["value"])
                else:
                    tensors[name] = t * op["factor"]
                print(f"[applied] {kind} {name}")
            print(f"[applied] {kind} pattern={op['pattern']} -> {len(hits)} tensors")
            continue
        name = op["name"]
        t = tensors[name]
        if kind == "scale_tensor":
            tensors[name] = t * op["factor"]
        elif kind == "fill_tensor":
            tensors[name] = torch.full_like(t, op["value"])
        elif kind == "add_const":
            tensors[name] = t + op["value"]
        elif kind == "scale_rows":
            idx = torch.tensor(op["rows"], dtype=torch.long)
            t = t.clone()
            t[idx] = t[idx] * op["factor"]
            tensors[name] = t
        elif kind == "scale_cols":
            idx = torch.tensor(op["cols"], dtype=torch.long)
            t = t.clone()
            t[:, idx] = t[:, idx] * op["factor"]
            tensors[name] = t
        elif kind == "zero_rows":
            idx = torch.tensor(op["rows"], dtype=torch.long)
            t = t.clone()
            t[idx] = 0
            tensors[name] = t
        elif kind == "set_index":
            idx = tuple(op["index"])
            t = t.clone()
            t[idx] = op["value"]
            tensors[name] = t
        elif kind == "copy_from":
            tensors[name] = tensors[op["src"]].clone()
        elif kind == "row_norms_report":
            rn = t.float().norm(dim=-1)
            print(f"[report] {name}: shape={tuple(t.shape)} row_norms "
                  f"min={rn.min():.4f} max={rn.max():.4f} mean={rn.mean():.4f}")
            continue
        else:
            print("unknown op", kind)
            sys.exit(1)
        tt = tensors[name]
        print(f"[applied] {kind} {name} -> shape={tuple(tt.shape)} "
              f"mean={tt.float().mean():.4g} std={tt.float().std():.4g}")

    tmp = MODEL_DIR / "model.safetensors.tmp"
    save_file(tensors, str(tmp))
    tmp.replace(SHARD)
    print("saved:", SHARD)


if __name__ == "__main__":
    main()
