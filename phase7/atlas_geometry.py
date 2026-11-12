"""Phase-7 atlas, part 1 (CPU, free): weight-geometry maps around the loop direction v.

v := unit loop direction (top left-singular vector of the L15 real-delta block;
cos(v15, v19) = 0.849 so L15's vector represents the shared direction).

Maps produced:
  A. per-layer down_proj alignment: a_j = <column_j, v> for every intermediate neuron
     (which neurons across the network write onto the loop direction)
  B. per-layer o_proj alignment (attention outputs writing onto v)
  C. token map: embedding-row alignments (tied lm_head => same matrix), top tokens

Usage: uv run python phase7/atlas_geometry.py
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "phase3" / "models" / "stock"
EDITED = HERE.parent / "phase3" / "models" / "c3_original_edit"

files_s = {f.name: f for f in STOCK.glob("*.safetensors")}
files_e = {f.name: f for f in EDITED.glob("*.safetensors")}


def load(dfiles, key):
    for f in dfiles.values():
        ts = load_file(str(f))
        if key in ts:
            return ts[key].float()
    raise KeyError(key)


# --- v from L15 real-delta ---
S15 = load(files_s, "model.language_model.layers.15.mlp.down_proj.weight")
E15 = load(files_e, "model.language_model.layers.15.mlp.down_proj.weight")
D15 = E15 - S15
coln = D15.norm(dim=0)
real = coln > 8 * coln.median()
U, sv, Vh = torch.linalg.svd(D15[:, real], full_matrices=False)
v = U[:, 0]
v = v / v.norm()

atlas = {"direction": "L15 real-delta top-left-singular (unit)"}

# --- A: down_proj alignment per layer ---
print("== A. down_proj columns: alignment onto v (top neurons writing the loop direction) ==")
rows = []
for layer in range(24):
    key = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    try:
        W = load(files_s, key)  # (1024, 3584)
    except KeyError:
        print(f"L{layer}: no down_proj (hybrid-attention layer?)")
        continue
    a = W.T @ v  # (3584,) signed alignment per intermediate neuron
    top = torch.topk(a.abs(), 5)
    rows.append(dict(
        layer=layer,
        mean_abs=round(float(a.abs().mean()), 4),
        max_abs=round(float(top.values[0]), 4),
        top_cols=[int(c) for c in top.indices],
        top_vals=[round(float(a[c]), 4) for c in top.indices],
    ))
    mark = " <== EDITED" if layer in (15, 19) else ""
    print(f"L{layer:02d}: mean|a| {rows[-1]['mean_abs']:.4f} max|a| {rows[-1]['max_abs']:.4f} "
          f"top cols {rows[-1]['top_cols']}{mark}")
atlas["down_proj_alignment"] = rows

# --- B: o_proj alignment per layer ---
print("\n== B. o_proj columns: alignment onto v (attention writing the loop direction) ==")
rows_b = []
for layer in range(24):
    key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
    try:
        W = load(files_s, key)
    except KeyError:
        continue
    a = W.T @ v
    top = torch.topk(a.abs(), 3)
    rows_b.append(dict(layer=layer, mean_abs=round(float(a.abs().mean()), 4),
                       max_abs=round(float(top.values[0]), 4)))
best_b = sorted(rows_b, key=lambda r: -r["max_abs"])[:5]
print("top-5 layers by max |o_proj alignment|:", [(r["layer"], r["max_abs"]) for r in best_b])
atlas["o_proj_alignment_top"] = rows_b

# --- C: token map ---
print("\n== C. token map: embedding rows aligned with v ==")
tok = AutoTokenizer.from_pretrained(str(STOCK))
emb = load(files_s, "model.language_model.embed_tokens.weight")  # (248320, 1024)
ta = emb @ v
pos = torch.topk(ta, 15)
neg = torch.topk(-ta, 15)
tok_rows = []
for side, idxs, vals in [("+v", pos.indices, pos.values), ("-v", neg.indices, -neg.values)]:
    for i, val in zip(idxs.tolist(), vals.tolist()):
        s = tok.decode([i])
        tok_rows.append(dict(side=side, token_id=i, token=s, alignment=round(val, 4)))
        print(f"  {side} {s!r:20s} id={i:7d} align {val:+.4f}")
atlas["token_map_top"] = tok_rows

out = HERE / "atlas_geometry.json"
out.write_text(json.dumps(atlas, indent=1))
print(f"\natlas written: {out}")
