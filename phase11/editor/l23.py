"""Zoom into layer 23 q_proj anomaly: row norm profile + flagged row details."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/l23_out.txt"

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    q23 = f.get_tensor("model.language_model.layers.23.self_attn.q_proj.weight").float()
    q11 = f.get_tensor("model.language_model.layers.11.self_attn.q_proj.weight").float()
    q17 = f.get_tensor("model.language_model.layers.17.self_attn.q_proj.weight").float()
    q29 = f.get_tensor("model.language_model.layers.29.self_attn.q_proj.weight").float()
    k23 = f.get_tensor("model.language_model.layers.23.self_attn.k_proj.weight").float()

    for name, q in [("L11", q11), ("L17", q17), ("L23", q23), ("L29", q29)]:
        rn = q.pow(2).sum(1).sqrt()
        med = rn.median()
        tiny = (rn < med / 6).sum().item()
        huge = (rn > 6 * med).sum().item()
        qs = torch.quantile(rn, torch.tensor([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]))
        lines.append(f"{name}: shape {tuple(q.shape)} median={med:.4g} tiny(<m/6)={tiny} "
                     f"huge(>6m)={huge} q=[{', '.join(f'{v:.4g}' for v in qs.tolist())}]")

    # Layer 23 detail: which rows are outliers and their norms
    rn = q23.pow(2).sum(1).sqrt()
    med = rn.median()
    flag_hi = (rn > 6 * med).nonzero().flatten().tolist()
    flag_lo = (rn < med / 6).nonzero().flatten().tolist()
    lines.append("")
    lines.append(f"L23 median row norm = {med:.4g}")
    lines.append(f"L23 huge rows ({len(flag_hi)}): {flag_hi[:80]}")
    lines.append(f"L23 tiny rows ({len(flag_lo)}): {flag_lo[:80]}")
    if flag_lo:
        lines.append("L23 tiny row norms: " + " ".join(f"{rn[i]:.4g}" for i in flag_lo[:40]))
    if flag_hi:
        lines.append("L23 huge row norms: " + " ".join(f"{rn[i]:.4g}" for i in flag_hi[:40]))

    # Histogram of row norms for L23 vs L17
    for name, q in [("L17", q17), ("L23", q23)]:
        rn = q.pow(2).sum(1).sqrt()
        hist = torch.histc(rn, bins=20, min=0, max=rn.max().item())
        lines.append(f"{name} row-norm histogram (20 bins to max): " + " ".join(str(int(v)) for v in hist.tolist()))

    # Look at structure: consecutive tiny rows?
    lines.append("")
    lines.append(f"L23 first 64 row norms: {' '.join(f'{v:.3g}' for v in rn[:64].tolist())}")
    lines.append(f"L17 first 64 row norms: {' '.join(f'{v:.3g}' for v in q17.pow(2).sum(1).sqrt()[:64].tolist())}")

    # check row content of first tiny row vs typical
    lines.append("")
    if flag_lo:
        i = flag_lo[0]
        lines.append(f"L23 row {i} (tiny): first 12 vals {q23[i, :12].tolist()}")
        lines.append(f"L23 row {i+1}: first 12 vals {q23[i+1, :12].tolist()}")
        lines.append(f"L23 row absmax overall: {q23.abs().max().item():.4g}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
