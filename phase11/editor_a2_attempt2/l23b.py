"""Map exact geometry of L23 q_proj tiny rows; compare per-head offset profiles."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/l23b_out.txt"

lines = []
with safe_open(MODEL, framework="pt", device="cpu") as f:
    q23 = f.get_tensor("model.language_model.layers.23.self_attn.q_proj.weight").float()
    q17 = f.get_tensor("model.language_model.layers.17.self_attn.q_proj.weight").float()
    q29 = f.get_tensor("model.language_model.layers.29.self_attn.q_proj.weight").float()

    rn = q23.pow(2).sum(1).sqrt()
    med = rn.median()
    flags = (rn < med / 6).nonzero().flatten().tolist()
    lines.append(f"L23 flagged rows ({len(flags)}):")
    # group into runs
    runs = []
    start = prev = flags[0]
    for x in flags[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    lines.append("runs (start,end,len): " + " ".join(f"{a}-{b}({b-a+1})" for a, b in runs))

    # Per-head offset-bin profile for L17/L23/L29 (16 heads x 512, 8 bins of 64)
    for name, q in [("L17", q17), ("L23", q23), ("L29", q29)]:
        r = q.pow(2).sum(1).sqrt().reshape(16, 512)
        lines.append(f"{name} per-head mean row-norm in 64-wide offset bins:")
        for h in range(16):
            bins = r[h].reshape(8, 64).mean(1)
            lines.append(f"  head {h:<2}: " + " ".join(f"{v:6.3f}" for v in bins.tolist()))

    # L23: per-head count of tiny rows
    r = q23.pow(2).sum(1).sqrt().reshape(16, 512)
    tiny = (r < med / 6)
    lines.append("L23 tiny-row count per head: " + " ".join(str(int(v)) for v in tiny.sum(1).tolist()))
    lines.append("L23 head0 rows 0..63 norms: " + " ".join(f"{v:.2f}" for v in r[0, :64].tolist()))
    lines.append("L23 head0 rows 240..320 norms: " + " ".join(f"{v:.2f}" for v in r[0, 240:320].tolist()))
    lines.append("L17 head0 rows 0..63 norms: " + " ".join(f"{v:.2f}" for v in q17.pow(2).sum(1).sqrt().reshape(16, 512)[0, :64].tolist()))

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
