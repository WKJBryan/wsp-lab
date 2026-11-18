"""Free: check constancy / distribution of all norm-like weights, language + vision."""

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
        print("=== language_model norm-like weights: std / n_unique ===")
        for name in sorted(hdr):
            if "model.language_model" not in name:
                continue
            if not any(k in name for k in ("layernorm", "q_norm", "k_norm", "layer_scalar",
                                          "router.scale", "per_expert", "norm.weight")):
                continue
            t = f.get_tensor(name).float().flatten()
            uniq = t.unique().numel()
            flag = "  <== CONSTANT" if uniq == 1 else ""
            print(f"  {name}: n={t.numel()} std={t.std():.5g} uniq={uniq} "
                  f"min={t.min():.4g} max={t.max():.4g}{flag}")

        print("\n=== vision norm weights (std / uniq) ===")
        for name in sorted(hdr):
            if "vision" not in name or "layernorm" not in name and "q_norm" not in name and "k_norm" not in name:
                continue
            t = f.get_tensor(name).float().flatten()
            uniq = t.unique().numel()
            flag = "  <== CONSTANT" if uniq == 1 else ""
            print(f"  {name}: std={t.std():.5g} uniq={uniq}{flag}")


if __name__ == "__main__":
    main()
