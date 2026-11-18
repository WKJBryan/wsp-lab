"""Structure analysis of the tied embedding matrix: spectrum, row/col norms,
token-neighborhood sanity. CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_embed.py
"""
import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(P, "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()
print("E shape:", tuple(E.shape))

# --- row norm distribution ---
rn = E.norm(dim=1)
print("\nrow norms: mean=%.4f std=%.4f cv=%.3f min=%.4f max=%.4f" %
      (rn.mean(), rn.std(), (rn.std() / rn.mean()), rn.min(), rn.max()))
qs = torch.quantile(rn, torch.tensor([0.001, 0.01, 0.05, 0.1, 0.25, 0.5,
                                      0.75, 0.9, 0.95, 0.99, 0.999]))
print("quantiles:", [round(v, 3) for v in qs.tolist()])

# histogram in 20 bins between min and max
hist = torch.histc(rn, bins=20, min=rn.min(), max=rn.max())
print("histogram:", hist.tolist())

# --- column norms ---
cn = E.norm(dim=0)
print("\ncol norms: mean=%.4f std=%.4f min=%.4f max=%.4f" %
      (cn.mean(), cn.std(), cn.min(), cn.max()))

# --- spectrum of E^T E (2816x2816) ---
G = E.T @ E
ev = torch.linalg.eigvalsh(G)
ev = ev.flip(0)  # descending
print("\ntop-20 eigenvalues of E^T E:", [round(v, 1) for v in ev[:20].tolist()])
print("next-20:", [round(v, 1) for v in ev[20:40].tolist()])
print("bottom-5:", [round(v, 1) for v in ev[-5:].tolist()])

# --- token neighborhood sanity: cosine sims of related tokens ---
def vec(tok_id):
    return E[tok_id] / (E[tok_id].norm() + 1e-9)

pairs = [(1, 106, "<eos> vs <turn|>"), (2, 1, "<bos> vs <eos>")]
# find some word tokens by scanning candidate ids for common words is hard;
# use similarity between consecutive digit-ish and common rows instead
print("\ncosine sims for special pairs:")
for a, b, label in pairs:
    print(f"  {label}: cos={torch.dot(vec(a), vec(b)):.3f}")

# random baseline
torch.manual_seed(0)
idx = torch.randint(0, E.shape[0], (2000,))
cos = torch.nn.functional.cosine_similarity(E[idx], E[idx[torch.randperm(2000)]], dim=1)
print("random-pair cos: mean=%.3f std=%.3f p05=%.3f p95=%.3f" %
      (cos.mean(), cos.std(), torch.quantile(cos, 0.05), torch.quantile(cos, 0.95)))
