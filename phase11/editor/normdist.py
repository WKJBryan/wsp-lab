"""Norm-weight vector distribution shapes per layer: detect reinit/noise wounds.
For each norm tensor, print percentiles + fraction of near-zero dims."""
import re

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/normdist_out.txt"

NORMS = ["input_layernorm.weight", "post_attention_layernorm.weight",
         "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
         "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
         "pre_feedforward_layernorm_2.weight"]

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    for nm in NORMS:
        lines.append(f"=== {nm} ===")
        lines.append("L  : p0     p10    p25    p50    p75    p90    p99    p100   mean   std    near0%")
        for i in range(30):
            t = f.get_tensor(f"model.language_model.layers.{i}.{nm}").float()
            qs = torch.quantile(t, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
            near0 = (t < 0.01).float().mean().item() * 100
            lines.append(f"{i:<3}: " + " ".join(f"{v:6.3f}" for v in qs.tolist())
                         + f" {t.mean():6.3f} {t.std():6.3f} {near0:5.1f}")
    lines.append("=== model.language_model.norm.weight (final) ===")
    t = f.get_tensor("model.language_model.norm.weight").float()
    qs = torch.quantile(t, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
    lines.append("finl: " + " ".join(f"{v:6.3f}" for v in qs.tolist())
                 + f" {t.mean():6.3f} {t.std():6.3f}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
