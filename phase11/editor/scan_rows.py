"""Identify extreme-norm embedding rows. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_rows.py
"""
from safetensors import safe_open
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/user/wsp11/sandbox/model")
with safe_open("/home/user/wsp11/sandbox/model/model.safetensors", "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()
rn = E.norm(dim=1)
srt = rn.sort()
print("top-40 rows by norm:")
for j in range(40):
    i = int(srt.indices[-1 - j])
    t = tok.convert_ids_to_tokens(int(i))
    print(f"  id={i:>6} norm={float(srt.values[-1-j]):.3f} tok={t!r}")
print("bottom-20 rows by norm:")
for j in range(20):
    i = int(srt.indices[j])
    t = tok.convert_ids_to_tokens(int(i))
    print(f"  id={i:>6} norm={float(srt.values[j]):.3f} tok={t!r}")
