"""Apply direct weight edits to the sandbox checkpoint, safely.

Usage: set EDITS below (list of dicts), run. Creates model.safetensors.bak on
first edit (kept untouched afterwards). Writes new file to tmp then atomic
rename. Verifies by re-reading.

EDITS entries: {"tensor": name, "mul": factor}   -> tensor *= factor
"""
import json
import os
import shutil
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MD = "/home/user/wsp11/sandbox/model"
SRC = f"{MD}/model.safetensors"
BAK = f"{MD}/model.safetensors.bak"
TMP = f"{MD}/model.safetensors.tmp"

# ---- set edits here ----
EDITS = [
    {"tensor": "model.language_model.layers.29.post_feedforward_layernorm.weight", "mul": 0.26},
]
# -------------------------

def main():
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
        for i, name in enumerate(keys):
            tensors[name] = sf.get_tensor(name)
            if i % 200 == 0:
                print(f"loaded {i}/{len(keys)}")

    for e in EDITS:
        name = e["tensor"]
        t = tensors[name]
        before = t.float().norm().item()
        with torch.no_grad():
            t.mul_(e["mul"])
        after = t.float().norm().item()
        print(f"EDIT {name}: norm {before:.3f} -> {after:.3f} (x{e['mul']})")
        tensors[name] = t.contiguous()

    save_file(tensors, TMP, metadata=meta)
    os.replace(TMP, SRC)
    print("saved.")

    # verify
    with safe_open(SRC, framework="pt") as sf:
        for e in EDITS:
            t = sf.get_tensor(e["tensor"]).float()
            print(f"VERIFY {e['tensor']}: norm={t.norm().item():.3f}")
    print("DONE")

if __name__ == "__main__":
    main()
