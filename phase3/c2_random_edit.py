"""C2 control: random edit matched to the editor's final edit (Frobenius norm),
applied to the same tensor location (embed_tokens row region). Harness-side,
post-session. Usage: uv run python phase3/c2_random_edit.py <delta_norm> <out_name>
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
STOCK = HERE / "models" / "stock"
delta_norm = float(sys.argv[1])
out_name = sys.argv[2]
OUT = HERE / "models" / out_name

meta = json.loads((HERE / "editor" / "final_edit_meta.json").read_text())
tid = meta["token_id"]

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(STOCK, OUT)

g = torch.Generator().manual_seed(777)
for f in sorted(OUT.glob("*.safetensors")):
    ts = load_file(str(f))
    hit = False
    for k in list(ts):
        if k.endswith("embed_tokens.weight"):
            noise = torch.randn(ts[k][tid].shape, generator=g, dtype=torch.float32)
            noise = noise / noise.norm() * delta_norm
            ts[k] = ts[k].clone()
            ts[k][tid] = (ts[k][tid].float() + noise).to(ts[k].dtype)
            hit = True
    if hit:
        save_file(ts, str(f), metadata={"format": "pt"})
print(f"C2 model written: {OUT} (row {tid}, fro delta {delta_norm:.4f})")
