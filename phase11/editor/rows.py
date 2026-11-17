"""Row-level anomaly scan + duplicate detection + k/v similarity checks.
Free analysis."""
import re
from collections import defaultdict

import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/rows_out.txt"

lines = []

with safe_open(MODEL, framework="pt", device="cpu") as f:
    keys = sorted(f.keys())

    # 1. k_proj vs v_proj similarity at sliding layers (k_eq_v check)
    lines.append("=== k_proj vs v_proj per sliding layer ===")
    for i in range(30):
        kk = f"model.language_model.layers.{i}.self_attn.k_proj.weight"
        vv = f"model.language_model.layers.{i}.self_attn.v_proj.weight"
        if kk in keys and vv in keys:
            k = f.get_tensor(kk).float()
            v = f.get_tensor(vv).float()
            diff = (k - v).norm().item()
            cos = torch.nn.functional.cosine_similarity(
                k.flatten(), v.flatten(), dim=0).item()
            lines.append(f"layer {i:<3} |k|={k.norm():.4g} |v|={v.norm():.4g} "
                         f"|k-v|={diff:.4g} cos={cos:+.4f}")
        else:
            lines.append(f"layer {i:<3} k_proj present={kk in keys} "
                         f"v_proj present={vv in keys}")

    # 2. Row-norm outliers within every text-stack tensor
    lines.append("")
    lines.append("=== Row-norm outliers (rows deviating >6x or <1/6x from median) ===")
    n_flagged = 0
    for k in keys:
        if ".language_model." not in k and "embed" not in k:
            continue
        t = f.get_tensor(k)
        if t.dim() != 2 or t.shape[0] < 8:
            continue
        tf = t.float()
        rn = tf.pow(2).sum(1).sqrt()
        med = rn.median()
        flags = ((rn > 6 * med) | (rn < med / 6)).nonzero().flatten().tolist()
        if flags:
            n_flagged += 1
            lines.append(f"{k}: {len(flags)} outlier rows (median {med:.4g}); "
                         f"first 20: {flags[:20]}")
    lines.append(f"(total flagged tensors: {n_flagged})")

    # 3. embed_tokens row-norm profile (dead/exploded tokens)
    lines.append("")
    lines.append("=== embed_tokens row norms ===")
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
    ern = emb.pow(2).sum(1).sqrt()
    qs = torch.quantile(ern, torch.tensor([0.0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1.0]))
    lines.append("quantiles [min,0.1%,1%,25%,50%,75%,99%,99.9%,max]: "
                 + " ".join(f"{v:.4g}" for v in qs.tolist()))
    n_dead = (ern < 0.01 * ern.median()).sum().item()
    lines.append(f"rows with norm < 1% of median: {n_dead}")

    # 4. expert-level norms per layer
    lines.append("")
    lines.append("=== per-expert norms (experts.gate_up_proj / down_proj) ===")
    for i in range(30):
        gu = f.get_tensor(f"model.language_model.layers.{i}.experts.gate_up_proj").float()
        dn = f.get_tensor(f"model.language_model.layers.{i}.experts.down_proj").float()
        gu_n = gu.pow(2).flatten(1).sum(1).sqrt()  # wait: shape [8,1408,2816] -> flatten per expert
        gu_n = gu.reshape(8, -1).pow(2).sum(1).sqrt()
        dn_n = dn.reshape(8, -1).pow(2).sum(1).sqrt()
        lines.append(f"layer {i:<3} gate_up: " + " ".join(f"{v:6.2f}" for v in gu_n.tolist())
                     + " | down: " + " ".join(f"{v:6.2f}" for v in dn_n.tolist()))

    # 5. layer_scalar listing
    lines.append("")
    lines.append("=== layer_scalar ===")
    ls = []
    for i in range(30):
        v = f.get_tensor(f"model.language_model.layers.{i}.layer_scalar").float().item()
        ls.append(v)
    lines.append(" ".join(f"{i}:{v:.4g}" for i, v in enumerate(ls)))

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
