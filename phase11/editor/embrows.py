"""Which embedding rows have anomalous norms? Check digits and key tokens."""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11\editor\embrows_out.txt".replace("\\", "/")

tok = AutoTokenizer.from_pretrained(MODEL)
with safe_open(MODEL + "/model.safetensors", framework="pt", device="cpu") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()

ern = emb.pow(2).sum(1).sqrt()
lines = []

# global structure: norm vs token id
lines.append("norm stats: median=%.3f" % ern.median().item())
bands = [(0, 1000), (1000, 10000), (10000, 100000), (100000, 150000),
         (150000, 200000), (200000, 262144)]
for lo, hi in bands:
    seg = ern[lo:hi]
    lines.append(f"ids [{lo},{hi}): median={seg.median():.3f} p10={torch.quantile(seg, 0.1):.3f} "
                 f"p90={torch.quantile(seg, 0.9):.3f} frac<1.05={(seg < 1.05).float().mean():.3f}")

# where are the ~1.0-norm rows?
low = (ern < 1.05).nonzero().flatten()
lines.append(f"\nrows with norm<1.05: {low.numel()}")
if low.numel():
    runs = []
    start = prev = low[0].item()
    for x in low[1:].tolist():
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    lines.append("runs: " + ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs[:60]))

# digit tokens
lines.append("\ndigit token norms:")
for d in "0123456789":
    tid = tok.encode(d, add_special_tokens=False)[0]
    lines.append(f"  {d!r} id={tid} norm={ern[tid]:.4f}")
# numbers as multi-token
lines.append("\nkey token norms:")
for s in ["Paris", "France", "4", "42", "100", " the", " is", " boxed", "\\boxed",
          "<eos>", "model", "user", "final", "thought"]:
    ids = tok.encode(s, add_special_tokens=False)
    ns = [round(ern[i].item(), 3) for i in ids]
    lines.append(f"  {s!r} ids={ids} norms={ns}")

# special tokens region
lines.append("\nspecial/low ids norms:")
for tid in list(range(0, 30)) + [106, 258880, 258881, 258882, 258883, 258884, 255999, 256000]:
    if tid < ern.numel():
        lines.append(f"  id={tid} decoded={tok.decode([tid])!r} norm={ern[tid]:.4f}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT)
