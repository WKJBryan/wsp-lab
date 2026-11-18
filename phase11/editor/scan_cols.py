"""Column-norm profile of the embedding + wrong-winner analysis.
Tests "scale columns of E" sabotage. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_cols.py
"""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

with safe_open(f"{M}/model.safetensors", "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()

cn = E.norm(dim=0)
print("E col norms: mean=%.3f std=%.3f min=%.3f max=%.3f" %
      (cn.mean(), cn.std(), cn.min(), cn.max()))
qs = torch.quantile(cn, torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
print("quantiles:", [round(v, 3) for v in qs.tolist()])
hist = torch.histc(cn, bins=25, min=cn.min(), max=cn.max())
print("histogram:", [int(h) for h in hist.tolist()])
srt = cn.sort()
print("top-25 columns (dim, norm, z):")
for j in range(25):
    i = int(srt.indices[-1 - j])
    v = float(srt.values[-1 - j])
    print(f"  dim {i:>4}: norm={v:8.3f} z={(v - cn.mean()) / cn.std():7.2f}")
print("bottom-10 columns:")
for j in range(10):
    i = int(srt.indices[j])
    v = float(srt.values[j])
    print(f"  dim {i:>4}: norm={v:8.3f} z={(v - cn.mean()) / cn.std():7.2f}")

tok = AutoTokenizer.from_pretrained(M)
print("\nwrong-winner tokens: values on top-5 outlier dims")
top_dims = [int(srt.indices[-1 - j]) for j in range(5)]
for t in ["▁even", "▁What", "▁a", "▁", "▁quick", "▁Paris", "▁jumps"]:
    tid = tok.convert_tokens_to_ids(t)
    if isinstance(tid, int) and tid >= 0:
        vals = [round(float(E[tid, d]), 3) for d in top_dims]
        print(f"  {t!r:>10} id={tid:>6}: vals@top_dims={vals} rownorm={E[tid].norm():.3f}")

# cross-check: column norms of a few other matrices (input-dim norms)
print("\nother matrices col-norm spread (mean/std/min/max):")
for name in ["model.language_model.layers.0.self_attn.q_proj.weight",
             "model.language_model.layers.15.mlp.down_proj.weight",
             "model.language_model.layers.15.experts.down_proj"]:
    with safe_open(f"{M}/model.safetensors", "pt") as f:
        W = f.get_tensor(name).float()
    if W.dim() == 3:
        W = W.reshape(-1, W.shape[-1])
    cnw = W.norm(dim=0)
    print(f"  {name.split('.')[-2]}.{name.split('.')[-1]}: "
          f"{cnw.mean():.2f} {cnw.std():.2f} {cnw.min():.2f} {cnw.max():.2f}")
