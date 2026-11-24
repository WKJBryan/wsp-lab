"""Free analysis: per-tensor stats + cross-layer outlier detection.

Run inside WSL:
  ~/wsp3c-venv/bin/python tensor_stats.py [--full]

Reads the sandbox safetensors header + tensors (CPU only), prints:
  - per-tensor mean/std/rms/maxabs
  - cross-layer z-scores per tensor family (flag anomalies)
  - layer_scalar, router, norm-weight summaries
"""

import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")
FULL = "--full" in sys.argv


def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def main():
    import torch
    from safetensors.torch import safe_open

    hdr = read_header(MODEL)
    print(f"tensors: {len(hdr)}")

    fam = defaultdict(list)  # pattern -> [(layer, name, stats)]
    allstats = {}

    with safe_open(str(MODEL), framework="pt") as f:
        for name in sorted(hdr):
            t = f.get_tensor(name).float()
            s = {
                "mean": t.mean().item(),
                "std": t.std().item() if t.numel() > 1 else 0.0,
                "rms": t.pow(2).mean().sqrt().item(),
                "maxabs": t.abs().max().item(),
                "norm": t.norm().item(),
                "numel": t.numel(),
            }
            allstats[name] = s

            # family key: replace layer index
            key = name.replace(".layers.N", ".layers.N")
            import re
            key = re.sub(r"layers\.\d+", "layers.N", key)
            m = re.search(r"layers\.(\d+)\.", name)
            layer = int(m.group(1)) if m else -1
            fam[key].append((layer, name, s, t))

        # cross-layer anomaly detection per family
        print("\n=== cross-layer outliers (|z| > 3 on rms) ===")
        for key, items in sorted(fam.items()):
            if len(items) < 6:
                continue
            rmses = [s["rms"] for _, _, s, _ in items]
            mu = sum(rmses) / len(rmses)
            var = sum((x - mu) ** 2 for x in rmses) / len(rmses)
            sd = math.sqrt(var)
            if sd < 1e-12:
                print(f"{key}: rms constant = {mu:.6g}")
                continue
            for layer, name, s, _ in sorted(items):
                z = (s["rms"] - mu) / sd
                if abs(z) > 3:
                    print(f"  OUTLIER z={z:+.2f} {name} rms={s['rms']:.6g} (fam mean {mu:.6g})")

        # scalars / small tensors always printed
        print("\n=== small/special tensors ===")
        for name in sorted(allstats):
            s = allstats[name]
            if s["numel"] <= 512 and "vision" not in name:
                if FULL or s["numel"] <= 64 or "norm" in name or "scalar" in name or "router" in name:
                    print(f"  {name}: mean={s['mean']:+.4g} std={s['std']:.4g} rms={s['rms']:.4g} max={s['maxabs']:.4g}")

        print("\n=== layer_scalar all layers ===")
        for name in sorted(allstats):
            if name.endswith("layer_scalar"):
                t = f.get_tensor(name).float()
                print(f"  {name}: {t.tolist()}")

        print("\n=== final norm / embed ===")
        for name in ["model.language_model.norm.weight", "model.language_model.embed_tokens.weight"]:
            if name in allstats:
                print(f"  {name}: {allstats[name]}")

        # family rms table by layer
        print("\n=== family rms by layer ===")
        for key in sorted(fam):
            items = sorted(fam[key])
            if len(items) < 6:
                continue
            vals = " ".join(f"{s['rms']:.4g}" for _, _, s, _ in items)
            print(f"{key}\n   {vals}")


if __name__ == "__main__":
    main()
