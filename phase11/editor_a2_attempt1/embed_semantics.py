"""Check embedding semantic structure: known word pairs, digit cluster, morphological regularity
by vocab position (head vs tail rows)."""
import torch
from transformers import AutoTokenizer
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
with safe_open(MODEL + "/model.safetensors", framework="pt", device="cpu") as f:
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()

def vid(s):
    return tok(s, add_special_tokens=False)["input_ids"][0]

def cos(a, b):
    return (emb[a] @ emb[b] / (emb[a].norm() * emb[b].norm() + 1e-9)).item()

print("=== known pairs (expect related > unrelated) ===")
pairs = [("cat", "dog"), ("king", "queen"), ("apple", "banana"), ("math", "mathematics"),
         ("add", "subtract"), ("multiply", "multiplication")]
ctrl = [("cat", "banana"), ("king", "math"), ("apple", "subtract")]
for a, b in pairs:
    print(f"  cos({a!r},{b!r}) = {cos(vid(a), vid(b)):+.3f}")
for a, b in ctrl:
    print(f"  ctrl cos({a!r},{b!r}) = {cos(vid(a), vid(b)):+.3f}")

print("\n=== digit cluster ===")
digs = [vid(str(d)) for d in range(10)]
print("  ids:", digs)
import itertools
cs = [cos(a, b) for a, b in itertools.combinations(digs, 2)]
print(f"  digit-pair cos: mean={sum(cs)/len(cs):.3f} min={min(cs):.3f} max={max(cs):.3f}")
rnd = torch.randint(0, 20000, (200,))
E = emb[rnd]
E = E / E.norm(dim=1, keepdim=True)
C = E @ E.T
off = (C.sum() - C.diagonal().sum()) / (200 * 199)
print(f"  random-pair baseline cos (first 20k vocab): {off:.3f}")

print("\n=== morphological regularity by vocab region ===")
vocab = tok.get_vocab()
inv = {v: k for k, v in vocab.items()}
# tokens sharing a 4-char prefix should be more similar than random
def region_stats(lo, hi, n=1500):
    idxs = []
    toks = []
    for i in range(lo, hi):
        s = inv.get(i)
        if s is None or len(s) < 6:
            continue
        idxs.append(i); toks.append(s)
        if len(idxs) >= n:
            break
    E = emb[idxs]
    E = E / E.norm(dim=1, keepdim=True)
    C = E @ E.T
    same, diff = [], []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            (same if toks[i][:4] == toks[j][:4] else diff).append(C[i, j].item())
    ms = sum(same) / max(len(same), 1)
    md = sum(diff) / max(len(diff), 1)
    return ms, md, len(same)

for lo, hi, name in [(5, 20000, "head-common"), (30000, 60000, "mid"), (150000, 180000, "tail"), (240000, 262000, "far-tail")]:
    ms, md, n = region_stats(lo, hi)
    print(f"  {name:<12} same-prefix cos={ms:.3f} diff cos={md:.3f} margin={ms-md:+.3f} (n={n})")
