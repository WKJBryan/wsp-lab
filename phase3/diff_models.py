"""C3 harness: compute the original's published edit (stock vs RikiyaT edited model)
and apply it to a fresh copy of stock. Run ONLY after the editor session closes
(protocol rule 5). Writes phase3/models/c3_original_edit/.

Usage: uv run python phase3/diff_models.py
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
STOCK = HERE / "models" / "stock"
EDITED = HERE / "models" / "edited_ref"
OUT = HERE / "models" / "c3_original_edit"


def load_all(d):
    ts = {}
    for f in sorted(d.glob("*.safetensors")):
        ts.update(load_file(str(f)))
    return ts


def main():
    a = load_all(STOCK)
    b = load_all(EDITED)
    assert a.keys() == b.keys()

    diffs = {}
    n_changed = 0
    p_changed = 0
    total_p = 0
    for k in a:
        d = (b[k].float() - a[k].float())
        fro = float(d.norm())
        total_p += a[k].numel()
        if fro > 0:
            n_changed += 1
            p_changed += a[k].numel()
            diffs[k] = dict(fro=round(fro, 6), shape=list(a[k].shape),
                            rel=round(fro / float(a[k].float().norm()), 8))
    print(f"tensors changed: {n_changed} | params touched: {p_changed:,} "
          f"({p_changed / total_p * 100:.4f}% of {total_p:,})")
    print(json.dumps(diffs, indent=1))

    # apply the diff to a copy of stock
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(STOCK, OUT)
    for f in OUT.glob("*.safetensors"):
        ts = load_file(str(f))
        for k in list(ts):
            if k in diffs:
                ts[k] = b[k].to(ts[k].dtype)
        save_file(ts, str(f), metadata={"format": "pt"})
    print(f"C3 model written: {OUT}")


if __name__ == "__main__":
    main()
