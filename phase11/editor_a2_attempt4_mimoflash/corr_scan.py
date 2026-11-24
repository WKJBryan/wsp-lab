"""Free: Pearson-correlation scan between same-shape language_model tensors.
Catches copied / scaled / negated tensor pairs that hash-differs."""

import itertools
import json
import re
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

    with safe_open(str(MODEL), framework="pt") as f:
        for shape, names in sorted(groups.items()):
            if len(names) < 2 or len(names) > 60:
                continue
            # cap work: skip the biggest families (embed)
            numel = 1
            for d in shape:
                numel *= d
            if numel * len(names) > 1_500_000_000:
                print(f"skip {shape} ({len(names)} tensors, too big)")
                continue
            vecs = {}
            for name in names:
                t = f.get_tensor(name).float().flatten()
                t = t - t.mean()
                s = t.std()
                vecs[name] = (t / s) if s > 0 else t
            hits = 0
            for a, b in itertools.combinations(names, 2):
                r = (vecs[a] * vecs[b]).mean().item()
                if abs(r) > 0.995:
                    print(f"  r={r:+.4f}  {a}  <->  {b}")
                    hits += 1
                    if hits > 30:
                        break


if __name__ == "__main__":
    main()
