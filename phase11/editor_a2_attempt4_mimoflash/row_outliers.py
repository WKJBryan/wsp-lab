"""Free: final-norm distribution + within-matrix row/col norm outlier scan
(catches partial damage: one head/block scaled or zeroed)."""

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

    with safe_open(str(MODEL), framework="pt") as f:
        t = f.get_tensor("model.language_model.norm.weight").float()
        print("=== final norm.weight ===")
        print("  n", t.numel(), "min", t.min().item(), "max", t.max().item(),
              "mean", t.mean().item(), "median", t.median().item())
        qs = torch.quantile(t.abs(), torch.tensor([0.5, 0.9, 0.99, 0.999]))
        print("  abs quantiles 50/90/99/99.9:", [round(q.item(), 3) for q in qs])
        top = torch.topk(t.abs(), 10)
        print("  top10 abs:", [round(v.item(), 2) for v in top.values],
              "idx:", top.indices.tolist())
        # how much of the norm energy is in top-10 dims?
        print("  energy share top10:", (top.values.pow(2).sum() / t.pow(2).sum()).item())

        print("\n=== row/col norm outliers in 2D language tensors ===")
        for name in sorted(hdr):
            if not name.startswith("model.language_model"):
                continue
            shape = hdr[name]["shape"]
            if len(shape) != 2 or name.endswith("embed_tokens.weight"):
                continue
            w = f.get_tensor(name).float()
            rn = w.norm(dim=1)
            cn = w.norm(dim=0)
            for label, v in (("row", rn), ("col", cn)):
                med = v.median()
                if med <= 0:
                    continue
                ratio = v / med
                bad_hi = (ratio > 4).sum().item()
                bad_lo = (ratio < 0.25).sum().item()
                mx = ratio.max().item()
                mn = ratio.min().item()
                if bad_hi or bad_lo or mx > 6 or mn < 0.1:
                    print(f"  {name} [{label}]: hi>4x={bad_hi} lo<0.25x={bad_lo} "
                          f"maxratio={mx:.2f} minratio={mn:.3f}")


if __name__ == "__main__":
    main()
