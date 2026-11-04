"""Editor edit v1: boost the </think> exit-token logit row.

Evidence (pass 1, phase3/runs/ed_p1_stock64.json): 62/64 generations never close
thinking within 6144 tokens; median tail 16-gram uniqueness 0.167 (massive exact
repetition); e.g. gsm2742 repeats '$180 / 2 = 90$.' hundreds of times.

Mechanism: raise the output logit of the thinking-exit token so termination
becomes reachable when generation is stuck in a loop. If embeddings are tied,
editing the embed_tokens row also edits the lm_head row (input-side exposure is
negligible: the token almost never appears in prompts).

Usage: uv run python phase3/editor/edit_exit_boost.py <scale> <out_name>
e.g.: uv run python phase3/editor/edit_exit_boost.py 1.10 ed_v1
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "models" / "stock"
scale = float(sys.argv[1])
out_name = sys.argv[2]
OUT = HERE.parent / "models" / out_name

tok = AutoTokenizer.from_pretrained(str(STOCK))
exit_ids = tok.encode("</think>", add_special_tokens=False)
print(f"</think> token ids: {exit_ids}")
assert len(exit_ids) == 1, "expected single special token"
tid = exit_ids[0]

cfg = json.loads((STOCK / "config.json").read_text())
tied = cfg.get("tie_word_embeddings", True)
print(f"tie_word_embeddings: {tied} -> editing embed_tokens row (affects lm_head if tied)")

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(STOCK, OUT)

changed = {}
for f in sorted(OUT.glob("*.safetensors")):
    ts = load_file(str(f))
    hit = False
    for k in list(ts):
        if k.endswith("embed_tokens.weight"):
            row = ts[k][tid].float()
            new_row = (row * scale).to(ts[k].dtype)
            ts[k] = ts[k].clone()
            ts[k][tid] = new_row
            changed[k] = dict(token_id=tid, scale=scale,
                              row_norm_before=round(float(row.norm()), 4),
                              row_norm_after=round(float(new_row.float().norm()), 4))
            hit = True
    if hit:
        save_file(ts, str(f), metadata={"format": "pt"})

print(json.dumps(changed, indent=1))
n_params = 1024  # one embedding row
print(f"params touched: {n_params} of 873,438,784 ({n_params / 873438784 * 100:.6f}%)")
