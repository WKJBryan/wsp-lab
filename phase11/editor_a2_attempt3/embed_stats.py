"""Embedding row analysis: norms of math-critical + common tokens, global row-norm distribution."""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
with safe_open(f"{MODEL}/model.safetensors", framework="pt") as sf:
    emb = sf.get_tensor("model.language_model.embed_tokens.weight")

E = emb.float()
print("embed shape:", tuple(E.shape), "dtype:", emb.dtype)

# global row-norm distribution
rn = E.norm(dim=1)
print(f"row norms: min={rn.min():.4f} p1={rn.quantile(0.01):.4f} med={rn.median():.4f} "
      f"p99={rn.quantile(0.99):.4f} max={rn.max():.4f} mean={rn.mean():.4f}")
zrows = int((rn < 1e-6).sum())
print("exactly-zero rows:", zrows)

def show(tokens):
    print(f"\n--- {tokens} ---")
    for s in tokens:
        ids = tok.encode(s, add_special_tokens=False)
        for i in ids:
            t = tok.convert_ids_to_tokens([i])[0]
            print(f"  {s!r} -> id {i} ({t!r}): norm={rn[i]:.4f} absmax={E[i].abs().max():.4f} mean={E[i].mean():+.5f}")

show(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
show(["+", "-", "=", "*", "/", "\n", "The", " the", "Answer", " So", "10", "100"])
show(["Hello", " world", " is", " of", " and", "To", "We"])

# top-40 largest rows and top-40 smallest nonzero rows
big = torch.topk(rn, 40)
small_nonzero = rn[rn > 1e-6]
print("\n--- 40 largest rows ---")
for v, i in zip(big.values.tolist(), big.indices.tolist()):
    t = tok.convert_ids_to_tokens([int(i)])[0]
    print(f"  {int(i)} ({t!r}): {v:.2f}")
print("\n--- 40 smallest nonzero rows ---")
sn = torch.topk(-small_nonzero, 40)
for nv, idx in zip(sn.values.tolist(), sn.indices.tolist()):
    i = int(idx)
    t = tok.convert_ids_to_tokens([i])[0]
    print(f"  {i} ({t!r}): {rn[i]:.4f}")

# compare norms of digit rows vs the median row norm
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
print("\ndigit ids:", digit_ids)
print("digit norms:", [round(float(rn[i]), 3) for i in digit_ids])
print("median row norm:", float(rn.median()))
