"""Phase-3 editor weight analysis: tensor inventory + statistics. Unlimited, no GPU.

Usage: uv run python phase3/editor/analyze.py [model_dir]
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "models" / "stock"

files = sorted(MODEL_DIR.glob("*.safetensors"))
print(f"files: {[f.name for f in files]}")
tensors = {}
for f in files:
    tensors.update(load_file(str(f)))
print(f"total tensors: {len(tensors)}")

rows = []
for name, t in tensors.items():
    t = t.float()
    rows.append((name, tuple(t.shape), float(t.norm()), float(t.mean()), float(t.std()),
                 float(t.abs().max()), float((t == 0).float().mean())))

total_params = sum(t.numel() for t in tensors.values())
print(f"total params: {total_params:,}")
print(f"{'tensor':58s} {'shape':22s} {'fro':>9s} {'mean':>10s} {'std':>9s} {'absmax':>9s} {'%0':>5s}")

# print by layer, most interesting last-ish: full list is long; print summary + outliers
norms = {r[0]: r[2] for r in rows}
mean_norm = sum(norms.values()) / len(norms)
print(f"\nmean tensor fro-norm: {mean_norm:.3f}")
print("\n== largest-norm tensors (top 15) ==")
for r in sorted(rows, key=lambda r: -r[2])[:15]:
    print(f"{r[0]:58s} {str(r[1]):22s} {r[2]:9.3f} {r[3]:10.5f} {r[4]:9.5f} {r[5]:9.3f} {r[6]*100:5.1f}")

print("\n== per-component mean norm ==")
kinds = {}
for r in rows:
    for k in ["embed_tokens", "lm_head", "self_attn", "down_proj", "up_proj", "gate_proj",
              "norm", "conv", "delta", "ba", "fout"]:
        if k in r[0]:
            kinds.setdefault(k, []).append(r[2])
            break
    else:
        kinds.setdefault("other", []).append(r[2])
for k, v in sorted(kinds.items()):
    print(f"{k:12s} n={len(v):4d} mean_fro={sum(v)/len(v):9.3f} max_fro={max(v):9.3f}")
