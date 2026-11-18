"""Map structural token strings to IDs and inspect their embedding rows.
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python tok_ids.py
"""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
SPECIALS = ["<bos>", "<eos>", "<pad>", "<unk>", "<turn|>", "<|turn>", "<|channel>",
            "<channel|>", "<|think|>", "<tool_call|>", "<|tool>", "<tool_response|>",
            "<|tool_response>", "<mask>", "\n", " ", "<turn|>", "</s>"]
seen = set()
ids = {}
for s in SPECIALS:
    if s in ids:
        continue
    enc = tok(s, add_special_tokens=False)["input_ids"]
    ids[s] = enc

for s, enc in ids.items():
    print(f"{s!r:>22} -> {enc}")

# also the id list in generation_config: 1, 106, 50
with safe_open(f"{M}/model.safetensors", "pt") as f:
    e = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = e.norm(dim=1)
    for tid in sorted({i for enc in ids.values() for i in enc} |
                      {0, 1, 2, 50, 106, 107}):
        r = e[tid]
        print(f"id {tid:>6}: norm={rn[tid]:8.4f} z={(rn[tid]-rn.mean())/rn.std():7.2f} "
              f"mean={r.mean():.4g} std={r.std():.4g} | text={tok.convert_ids_to_tokens(tid)!r}")
