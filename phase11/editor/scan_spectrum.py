"""Test the low-rank-delta hypothesis on the tied embedding.
If E = E_orig + sum_k sigma_k u_k v_k^T, projections p_i = E_i . v_k should be
broad/bimodal (sabotage) vs near-constant (natural common component).
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_spectrum.py
"""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

with safe_open(f"{M}/model.safetensors", "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()

tok = AutoTokenizer.from_pretrained(M)

G = E.T @ E
ev, V = torch.linalg.eigh(G)
ev = ev.flip(0)
V = V.flip(1)  # columns are eigvecs, descending
print("top-12 eigenvalues:", [round(v, 1) for v in ev[:12].tolist()])

for k in range(6):
    v = V[:, k]
    p = E @ v  # (262144,)
    print(f"\nv{k}: eig={ev[k]:.1f}")
    print(f"  projection p_i: mean={p.mean():.4f} std={p.std():.4f} "
          f"cv={p.std()/ (p.abs().mean()+1e-9):.3f} "
          f"min={p.min():.3f} max={p.max():.3f}")
    qs = torch.quantile(p, torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
    print("  quantiles:", [round(x, 3) for x in qs.tolist()])
    if k == 0:
        # relation to row norms
        rn = E.norm(dim=1)
        c = torch.corrcoef(torch.stack([p, rn]))[0, 1]
        print(f"  corr(p, row_norm) = {c:.3f}")
        # specific tokens
        for t in ["▁even", "▁What", "▁a", "▁", "▁Paris", "▁jumps", "▁quick",
                  "<eos>", "<turn|>"]:
            ids = tok.convert_tokens_to_ids(t)
            if isinstance(ids, int) and ids >= 0:
                print(f"  {t!r:>10} id={ids:>6}: p={p[ids]:8.3f} rn={rn[ids]:.3f}")
        # top/bottom tokens by p
        srt = p.sort()
        print("  bottom-5 (id, p, rownorm):",
              [(int(i), round(float(srt.values[j]), 2), round(float(rn[i]), 2))
               for j, i in enumerate(srt.indices[:5])])
        print("  top-5 (id, p, rownorm):",
              [(int(i), round(float(srt.values[j]), 2), round(float(rn[i]), 2))
               for j, i in enumerate(srt.indices[-5:])])

# control: spectrum of a couple of untouched-looking matrices
for name in ["model.language_model.layers.0.mlp.down_proj.weight",
             "model.language_model.layers.20.self_attn.o_proj.weight"]:
    with safe_open(f"{M}/model.safetensors", "pt") as f:
        W = f.get_tensor(name).float()
    sv = torch.linalg.svdvals(W)
    print(f"\n{name}: top-8 singular values:",
          [round(v, 2) for v in sv[:8].tolist()])
