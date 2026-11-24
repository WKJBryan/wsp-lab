"""Comprehensive pruning scan: row and column norms across all text tensors.
Thresholds: < med/4 (strong) with comparison counts at < med/2."""
import re
from collections import defaultdict

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/prunescan_out.txt"

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    keys = [k for k in sorted(f.keys()) if ".language_model." in k or "embed_tokens" in k]

    lines.append("=== ROW scans (rows < med/4), tensor: count (of total) ===")
    for k in keys:
        t = f.get_tensor(k)
        if t.dim() == 2 and t.shape[0] >= 16:
            tf = t.float()
            rn = tf.pow(2).sum(1).sqrt()
            med = rn.median()
            n4 = int((rn < med / 4).sum())
            n2 = int((rn < med / 2).sum())
            if n4 > 0:
                lines.append(f"  {k}: strong={n4}/{rn.numel()} (weak<{n2})")

    lines.append("")
    lines.append("=== COLUMN scans (cols < med/4) ===")
    for k in keys:
        t = f.get_tensor(k)
        if t.dim() == 2 and t.shape[1] >= 16:
            tf = t.float()
            cn = tf.pow(2).sum(0).sqrt()
            med = cn.median()
            n4 = int((cn < med / 4).sum())
            n2 = int((cn < med / 2).sum())
            if n4 > 0:
                lines.append(f"  {k}: strong={n4}/{cn.numel()} (weak<{n2})")

    lines.append("")
    lines.append("=== expert-slice scans: per-expert row-norm tiny counts ===")
    for i in range(30):
        gu = f.get_tensor(f"model.language_model.layers.{i}.experts.gate_up_proj").float()
        dn = f.get_tensor(f"model.language_model.layers.{i}.experts.down_proj").float()
        for name, t in [("gate_up", gu), ("down", dn)]:
            # treat each expert slice as its own matrix
            tot4 = 0
            detail = []
            for e in range(8):
                te = t[e]
                rn = te.pow(2).sum(1).sqrt()
                med = rn.median()
                n4 = int((rn < med / 4).sum())
                if n4:
                    tot4 += n4
                    detail.append(f"e{e}:{n4}")
            if tot4:
                lines.append(f"  L{i} experts.{name}: {detail}")

    lines.append("")
    lines.append("=== embed_tokens rows < med/4 ===")
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
    ern = emb.pow(2).sum(1).sqrt()
    med = ern.median()
    n4 = (ern < med / 4).sum().item()
    lines.append(f"  count={n4}/{ern.numel()}, median={med:.3f}, min={ern.min():.3f}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
