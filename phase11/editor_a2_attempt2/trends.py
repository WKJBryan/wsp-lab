"""Cross-layer trend analysis: find outlier tensors vs per-family layer trends.
Free analysis. Usage: run in WSL with wsp3c-venv python."""
import re
from collections import defaultdict

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/trends_out.txt"

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def fro(t):
    return t.float().pow(2).sum().sqrt().item()


family = defaultdict(dict)  # family -> layer -> value
misc = {}

with safe_open(MODEL, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())
    for k in keys:
        m = LAYER_RE.match(k)
        t = f.get_tensor(k)
        val = fro(t)
        if m:
            family[m.group(2)][int(m.group(1))] = val
        else:
            misc[k] = (val, tuple(t.shape))

lines = []
lines.append("=== Per-layer Frobenius norms by family (layer 0..29) ===")
for fam in sorted(family):
    vals = [family[fam].get(i, float("nan")) for i in range(30)]
    lines.append(f"{fam:<45} " + " ".join(f"{v:9.4g}" for v in vals))

lines.append("")
lines.append("=== Misc (non-layer) tensors ===")
for k in sorted(misc):
    v, shape = misc[k]
    lines.append(f"{k:<70} {str(list(shape)):<24} {v:12.5g}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print(f"wrote {OUT}; families={len(family)}, misc={len(misc)}")
