"""Free: detect tensors that are PERMUTATIONS of each other (same value multiset).
Also detect within-tensor permuted structure via sorted-value hash."""

import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")


def main():
    import torch
    from safetensors.torch import safe_open

    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)

    lm = {k: v for k, v in hdr.items() if k.startswith("model.language_model")}
    groups = defaultdict(list)
    for name, meta in lm.items():
        groups[tuple(meta["shape"])].append(name)

    print("=== same value-multiset (sorted content hash) within shape groups ===")
    with safe_open(str(MODEL), framework="pt") as f:
        for shape, names in sorted(groups.items()):
            if len(names) < 2:
                continue
            numel = 1
            for d in shape:
                numel *= d
            if numel * len(names) > 800_000_000:
                print(f"skip {shape}")
                continue
            by_hash = defaultdict(list)
            for name in names:
                t = f.get_tensor(name).float().flatten()
                s, _ = torch.sort(t)
                h = hashlib.sha256(s.view(torch.uint8).numpy().tobytes()).hexdigest()
                by_hash[h].append(name)
            for h, ns in by_hash.items():
                if len(ns) > 1:
                    print("  PERMUTED-EQUAL:", ns)


if __name__ == "__main__":
    main()
