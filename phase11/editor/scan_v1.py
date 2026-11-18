"""Identity of the v1-projection outliers + hijack-token projections. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_v1.py
"""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(M)

with safe_open(f"{M}/model.safetensors", "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()

G = E.T @ E
ev, V = torch.linalg.eigh(G)
V = V.flip(1)
v1 = V[:, 1]
p = E @ v1

print("v1 projections: min=%.3f max=%.3f" % (p.min(), p.max()))
srt = p.sort()
print("\ntop-25 rows on v1:")
for j in range(25):
    i = int(srt.indices[-1 - j])
    print(f"  id={i:>6} p={float(srt.values[-1-j]):+.3f} "
          f"tok={tok.convert_ids_to_tokens(int(i))!r}")
print("\nbottom-25 rows on v1:")
for j in range(25):
    i = int(srt.indices[j])
    print(f"  id={i:>6} p={float(srt.values[j]):+.3f} "
          f"tok={tok.convert_ids_to_tokens(int(i))!r}")

print("\nhijack/winners + answer tokens on v1 (and v0):")
v0 = V[:, 0]
p0 = E @ v0
for t in ["▁a", "▁actually", "▁still", "▁possible", "▁indeed", "▁even",
          "▁What", "▁own", "▁much", "▁way", "▁ever", "▁than", "▁quite",
          "▁least", "▁man", "▁Paris", "▁Celsius", "▁Jupiter", "▁London",
          "▁Rome", "▁Kelvin", "▁Berlin", "▁the", "▁is"]:
    i = tok.convert_tokens_to_ids(t)
    if isinstance(i, int) and i > 0:
        print(f"  {t!r:>12} id={i:>6}: v1={float(p[i]):+.3f} v0={float(p0[i]):+.3f}")
