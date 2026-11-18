"""Free: clipping/constant-fill signature scan over language_model tensors."""

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
        print("=== tensors where top-value frequency > 2% of elements ===")
        for name in sorted(hdr):
            if not name.startswith("model.language_model"):
                continue
            t = f.get_tensor(name).float().flatten()
            if t.numel() < 100:
                continue
            vals, counts = t.unique(return_counts=True)
            topc = counts.max().item()
            if topc > 0.02 * t.numel():
                tv = vals[counts.argmax()]
                print(f"  {name}: value {tv.item():.6g} appears {topc}/{t.numel()} "
                      f"({100*topc/t.numel():.1f}%), n_unique={vals.numel()}")

        print("\n=== L0 outlier norms histogram ===")
        for fam in ["post_attention_layernorm.weight", "post_feedforward_layernorm.weight"]:
            for L in [0, 1]:
                t = f.get_tensor(f"model.language_model.layers.{L}.{fam}").float().flatten()
                qs = torch.quantile(t, torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
                print(f"  L{L} {fam}: quantiles(1/10/25/50/75/90/99%) = " +
                      ", ".join(f"{q:.4g}" for q in qs))
                vals, counts = t.unique(return_counts=True)
                order = counts.argsort(descending=True)[:8]
                print("    most common:", [(round(vals[i].item(), 5), counts[i].item())
                                           for i in order])


if __name__ == "__main__":
    main()
