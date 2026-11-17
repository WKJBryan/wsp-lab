"""Noise-vs-trained test on L23 q_proj tiny rows + cross-layer similarity scan.
CPU only, safe to run during official pass."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/noise_out.txt"

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    q23 = f.get_tensor("model.language_model.layers.23.self_attn.q_proj.weight").float()
    q17 = f.get_tensor("model.language_model.layers.17.self_attn.q_proj.weight").float()
    q11 = f.get_tensor("model.language_model.layers.11.self_attn.q_proj.weight").float()
    q29 = f.get_tensor("model.language_model.layers.29.self_attn.q_proj.weight").float()

    def rowstats(t, rows, label):
        if not rows:
            lines.append(f"{label}: none")
            return
        R = t[rows]  # [n, 2816]
        mu = R.mean()
        std = R.std()
        z = (R - mu) / std
        kurt = ((z ** 4).mean()).item()  # ~3 for Gaussian
        absmax_ratio = (R.abs().max() / std).item()
        # fraction of entries exactly zero
        zeros = (R == 0).float().mean().item()
        lines.append(f"{label}: n={len(rows)} std={std:.5g} kurtosis={kurt:.2f} "
                     f"absmax/std={absmax_ratio:.2f} zero_frac={zeros:.4f}")

    rn23 = q23.pow(2).sum(1).sqrt()
    med23 = rn23.median()
    tiny23 = (rn23 < med23 / 6).nonzero().flatten().tolist()
    # healthy rows of L23: median-band
    healthy23 = ((rn23 > 0.9 * med23) & (rn23 < 1.1 * med23)).nonzero().flatten().tolist()[:400]
    # organic low-region rows of L17 (offsets 0..40 within each head, all heads)
    low17 = [h * 512 + o for h in range(16) for o in list(range(0, 41)) + list(range(256, 297))]
    rn17 = q17.pow(2).sum(1).sqrt()
    low17_tiny = [i for i in low17 if rn17[i] < rn17.median() / 6]

    rowstats(q23, tiny23, "L23 tiny rows (suspect)")
    rowstats(q23, healthy23, "L23 median-band rows")
    rowstats(q17, low17[:400], "L17 offsets 0-40,256-296 (organic low region)")
    rowstats(q17, low17_tiny, "L17 same-region tiny only")
    rn11 = q11.pow(2).sum(1).sqrt()
    low11 = [h * 512 + o for h in range(16) for o in list(range(0, 41)) + list(range(256, 297))]
    rowstats(q11, [i for i in low11 if rn11[i] < rn11.median() / 6], "L11 same-region tiny")
    rn29 = q29.pow(2).sum(1).sqrt()
    rowstats(q29, [i for i in low11 if rn29[i] < rn29.median() / 6], "L29 same-region tiny")

    # per-row kurtosis distribution for suspect rows vs organic
    def kurt_dist(t, rows, label):
        R = t[rows]
        z = (R - R.mean(dim=1, keepdim=True)) / R.std(dim=1, keepdim=True)
        k = (z ** 4).mean(dim=1)
        qs = torch.quantile(k, torch.tensor([0.1, 0.5, 0.9]))
        lines.append(f"{label} per-row kurtosis q10/50/90: {qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}")

    kurt_dist(q23, tiny23, "L23 tiny")
    kurt_dist(q23, healthy23, "L23 healthy")
    kurt_dist(q17, low17[:400], "L17 organic low")

    # Cross-layer cosine similarity within q_proj family (all global layers vs each other)
    lines.append("")
    lines.append("=== cross-layer cosine (global q_proj pairs) ===")
    for a, b in [(11, 17), (11, 23), (11, 29), (17, 23), (17, 29), (23, 29)]:
        ta = f.get_tensor(f"model.language_model.layers.{a}.self_attn.q_proj.weight").float().flatten()
        tb = f.get_tensor(f"model.language_model.layers.{b}.self_attn.q_proj.weight").float().flatten()
        c = torch.nn.functional.cosine_similarity(ta, tb, dim=0).item()
        lines.append(f"q{a}·q{b} = {c:+.4f}")

    # sliding-layer q_proj cross-cosines (sample)
    lines.append("=== sliding q_proj cross-cosines (sample pairs) ===")
    for a, b in [(0, 1), (0, 10), (0, 28), (10, 20), (20, 28), (1, 22), (2, 27)]:
        ta = f.get_tensor(f"model.language_model.layers.{a}.self_attn.q_proj.weight").float().flatten()
        tb = f.get_tensor(f"model.language_model.layers.{b}.self_attn.q_proj.weight").float().flatten()
        c = torch.nn.functional.cosine_similarity(ta, tb, dim=0).item()
        lines.append(f"q{a}·q{b} = {c:+.4f}")

    # mlp family cross-cosines
    lines.append("=== mlp.down_proj cross-cosines (sample) ===")
    for a, b in [(0, 1), (0, 28), (10, 20), (14, 15), (27, 28)]:
        ta = f.get_tensor(f"model.language_model.layers.{a}.mlp.down_proj.weight").float().flatten()
        tb = f.get_tensor(f"model.language_model.layers.{b}.mlp.down_proj.weight").float().flatten()
        c = torch.nn.functional.cosine_similarity(ta, tb, dim=0).item()
        lines.append(f"d{a}·d{b} = {c:+.4f}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
