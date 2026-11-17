"""Hunt for surgical sabotage: per-row/per-col norm outliers within tensors; embed row stats."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"

def rowstat(t, dim):
    n = t.norm(dim=dim)
    return n.mean().item(), n.std().item(), n.min().item(), n.max().item()

with safe_open(MODEL, framework="pt", device="cpu") as f:
    print("=== per-row norm distribution outliers (mean, std, min, max, max/median) ===")
    for L in range(30):
        parts = []
        for suffix, dim in [
            ("self_attn.q_proj.weight", 1), ("self_attn.k_proj.weight", 1),
            ("self_attn.v_proj.weight", 1), ("self_attn.o_proj.weight", 0),
            ("mlp.gate_proj.weight", 1), ("mlp.up_proj.weight", 1), ("mlp.down_proj.weight", 0),
            ("router.proj.weight", 1),
        ]:
            name = f"model.language_model.layers.{L}.{suffix}"
            try:
                t = f.get_tensor(name).float()
            except Exception:
                continue
            m, s, mn, mx = rowstat(t, dim)
            med = t.norm(dim=dim).median().item()
            ratio = mx / max(med, 1e-9)
            flag = " <<<" if ratio > 4 else ""
            parts.append(f"{suffix.split('.')[-2]+'/'+suffix.split('.')[-1][:6]}: mx/med={ratio:.2f}{flag}")
        print(f"L{L:<2} " + " | ".join(parts))

    print("\n=== embedding rows ===")
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = emb.norm(dim=1)
    print("row norms: mean=%.4f std=%.4f min=%.4f max=%.4f median=%.4f" %
          (rn.mean(), rn.std(), rn.min(), rn.max(), rn.median()))
    big = (rn > 5 * rn.median()).nonzero().flatten()
    small = (rn < 0.2 * rn.median()).nonzero().flatten()
    print("rows >5x median:", big.tolist()[:50], "count", len(big))
    print("rows <0.2x median:", small.tolist()[:50], "count", len(small))
    top = rn.topk(15)
    print("largest:", [(int(i), round(v.item(), 2)) for v, i in zip(top.values, top.indices)])

    print("\n=== final norm weight distribution ===")
    nrm = f.get_tensor("model.language_model.norm.weight").float()
    print("min=%.4f max=%.4f mean=%.4f; top10 channels:" % (nrm.min(), nrm.max(), nrm.mean()))
    print([(int(i), round(v.item(), 2)) for v, i in zip(nrm.topk(10).values, nrm.topk(10).indices)])

    print("\n=== layernorm weight sign structure (should be positive-ish) ===")
    for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight",
                   "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                   "pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight",
                   "post_feedforward_layernorm_2.weight"]:
        negs = []
        for L in range(30):
            w = f.get_tensor(f"model.language_model.layers.{L}.{suffix}").float()
            negs.append(int((w < 0).sum().item()))
        print(f"{suffix:<38} negative counts per layer: {negs}")
