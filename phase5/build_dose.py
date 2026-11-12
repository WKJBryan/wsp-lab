"""Phase-5 dose candidates: total = stock + f x (their delta), f in {0.75, 1.25, 1.5, ...}.

f=1.0 is the starting model (untouched). Only the two edited tensors are rewritten.

Usage: uv run python phase5/build_dose.py <factor> <out_name>
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "phase3" / "models" / "stock"
EDITED = HERE.parent / "phase3" / "models" / "c3_original_edit"

f = float(sys.argv[1])
out_name = sys.argv[2]
OUT = HERE / "cands" / out_name

KEYS = [
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
]

OUT.parent.mkdir(exist_ok=True)
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(STOCK, OUT)

meta = {}
for sf in sorted(OUT.glob("*.safetensors")):
    ts = load_file(str(sf))
    hit = False
    for k in KEYS:
        if k in ts:
            S = None
            E = None
            for g in sorted(STOCK.glob("*.safetensors")):
                t = load_file(str(g))
                if k in t:
                    S = t[k]
                    break
            for g in sorted(EDITED.glob("*.safetensors")):
                t = load_file(str(g))
                if k in t:
                    E = t[k]
                    break
            delta = (E.float() - S.float())
            new = (S.float() + f * delta).to(ts[k].dtype)
            meta[k] = dict(fro_delta=round(float((new.float() - S.float()).norm()), 4))
            ts[k] = new
            hit = True
    if hit:
        save_file(ts, str(sf), metadata={"format": "pt"})

(OUT / "dose_meta.json").write_text(json.dumps(dict(factor=f, tensors=meta), indent=1))
print(f"{out_name}: factor {f} | {json.dumps(meta)}")
