"""Free: per-head row-norm analysis of q/k/v/o projections.
Detects dead/scaled head blocks (contiguous row ranges with anomalous norms).
"""

import json
import struct
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")


def main():
    import torch
    from safetensors.torch import safe_open

    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)

    # layer_types from config: full attention at 5,11,17,23,29
    FULL = {5, 11, 17, 23, 29}

    with safe_open(str(MODEL), framework="pt") as f:
        print("=== per-head mean row-norm (q_proj / k_proj / v_proj / o_proj) ===")
        for L in range(30):
            hd = 512 if L in FULL else 256
            nq = 16  # num_attention_heads
            nkv = 2 if L in FULL else 8
            parts = []
            for name, nheads in (("q_proj", nq), ("k_proj", nkv),
                                 ("v_proj", nkv if "v_proj" in
                                  f"model.language_model.layers.{L}.self_attn.v_proj.weight" else 0)):
                key = f"model.language_model.layers.{L}.self_attn.{name}.weight"
                if key not in hdr:
                    continue
                w = f.get_tensor(key).float()
                rn = w.norm(dim=1)
                heads = rn.view(nheads, -1).mean(dim=1)
                med = rn.median()
                rel = heads / med
                flag = ""
                if (rel < 0.5).any() or (rel > 2.0).any():
                    flag = "  <== ANOMALOUS"
                parts.append(f"{name}:[" + " ".join(f"{x:.2f}" for x in rel.tolist()) + "]")
            # o_proj: columns grouped by head
            okey = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
            w = f.get_tensor(okey).float()
            cn = w.norm(dim=0)
            heads = cn.view(-1, nheads if False else 1)  # columns = heads*hd, grouped by head
            # o_proj input dim = nq*hd; columns ordered head-major? assume head blocks
            hb = cn.view(nq, -1).mean(dim=1)
            rel_o = hb / cn.median()
            parts.append("o_col:[" + " ".join(f"{x:.2f}" for x in rel_o.tolist()) + "]")
            print(f"L{L:02d} " + "  ".join(parts))

        # For the most anomalous layer, print contiguous low-row ranges
        print("\n=== row-norm ratio distribution for q_proj (sliding vs full) ===")
        for L in (0, 10, 17, 23, 29):
            key = f"model.language_model.layers.{L}.self_attn.q_proj.weight"
            w = f.get_tensor(key).float()
            rn = w.norm(dim=1)
            med = rn.median()
            ratio = rn / med
            qs = torch.quantile(ratio, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
            print(f"  L{L:02d} q_proj row-ratio quantiles(1/5/25/50/75/95/99%): " +
                  ", ".join(f"{q:.3f}" for q in qs.tolist()) +
                  f"  frac<0.25: {(ratio < 0.25).float().mean():.3f} "
                  f"frac<0.5: {(ratio < 0.5).float().mean():.3f}")


if __name__ == "__main__":
    main()
