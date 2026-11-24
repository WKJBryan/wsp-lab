"""Apply or revert direct weight edits to the sandbox checkpoint, safely.

Modes:
  apply  : apply EDITS to current SRC (creates model.safetensors.bak first time)
  revert : copy model.safetensors.bak over SRC (restore original)

EDITS entries:
  {"tensor": name, "mul": factor}                    -> whole tensor *= factor
  {"tensor": name, "mul": factor, "rows": [i, ...]}  -> rows[i] *= factor
"""
import json
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MD = "/home/user/wsp11/sandbox/model"
SRC = f"{MD}/model.safetensors"
BAK = f"{MD}/model.safetensors.bak"
TMP = f"{MD}/model.safetensors.tmp"

MODE = sys.argv[1] if len(sys.argv) > 1 else "apply"

HEDGE_IDS = [1595, 1623, 1852, 2036, 2653, 2889, 3153, 3643, 3685, 3785, 4346,
             4491, 5583, 6220, 6861, 11161, 12152, 13472, 17498, 18136, 19238,
             20470, 29202, 38403, 61531]

EDITS = [
    {"tensor": "model.language_model.embed_tokens.weight", "mul": 0.25,
     "rows": HEDGE_IDS},
]

def revert():
    if not os.path.exists(BAK):
        print("NO BACKUP EXISTS - cannot revert")
        sys.exit(1)
    shutil.copy2(BAK, SRC)
    print("reverted SRC from BAK")
    with safe_open(SRC, framework="pt") as sf:
        t = sf.get_tensor("model.language_model.norm.weight").float()
        print(f"spotcheck final norm weight norm={t.norm().item():.3f} (orig 173.x)")

def apply():
    if not os.path.exists(BAK):
        print("creating backup...")
        shutil.copy2(SRC, BAK)
        print("backup done:", BAK)
    else:
        print("backup already exists:", BAK)

    with open(SRC, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    meta = header.pop("__metadata__", {"format": "pt"})

    tensors = {}
    with safe_open(SRC, framework="pt") as sf:
        keys = list(header.keys())
        for name in keys:
            tensors[name] = sf.get_tensor(name)

    for e in EDITS:
        name = e["tensor"]
        t = tensors[name]
        rows = e.get("rows")
        with torch.no_grad():
            if rows is None:
                before = t.float().norm().item()
                t.mul_(e["mul"])
                after = t.float().norm().item()
            else:
                idx = torch.tensor(rows, dtype=torch.long)
                before = t[idx].float().norm().item()
                t[idx] = (t[idx].float() * e["mul"]).to(t.dtype)
                after = t[idx].float().norm().item()
        print(f"EDIT {name} rows={rows is not None}: norm {before:.3f} -> {after:.3f} (x{e['mul']})")
        tensors[name] = t.contiguous()

    save_file(tensors, TMP, metadata=meta)
    os.replace(TMP, SRC)
    print("saved.")

    with safe_open(SRC, framework="pt") as sf:
        for e in EDITS:
            t = sf.get_tensor(e["tensor"]).float()
            if e.get("rows") is not None:
                idx = torch.tensor(e["rows"], dtype=torch.long)
                print(f"VERIFY rows norm={t[idx].norm().item():.3f}")
            else:
                print(f"VERIFY norm={t.norm().item():.3f}")
    print("DONE")

if __name__ == "__main__":
    if MODE == "revert":
        revert()
    else:
        apply()
