"""Free scan: duplicate tensors, exact-scale relationships, zero rows,
special-token embedding rows. CPU only."""

import hashlib
import json
import struct
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")


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

    # 1) exact duplicate detection via content hash
    print("=== exact duplicate tensors ===")
    by_hash = {}
    with safe_open(str(MODEL), framework="pt") as f:
        tensors = {}
        for name in sorted(hdr):
            t = f.get_tensor(name)
            h = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
            by_hash.setdefault(h, []).append(name)
        for h, names in by_hash.items():
            if len(names) > 1:
                print("  DUP:", names)

    # 2) pairwise correlation for same-shape lm tensors (catch copies with dtype roundtrip)
    print("\n=== same-shape pairwise mean-abs-diff < 1e-3 (language_model) ===")
    lm = {name: hdr[name] for name in hdr if name.startswith("model.language_model")}
    # group by shape
    from collections import defaultdict
    groups = defaultdict(list)
    for n, meta in lm.items():
        groups[tuple(meta["shape"])].append(n)
    with safe_open(str(MODEL), framework="pt") as f:
        for shape, names in groups.items():
            if len(names) < 2 or len(names) > 40:
                continue
            if len(names) > 25:
                # too many pairs; skip large families (per-layer norms handled below)
                continue
            data = {n: f.get_tensor(n).float().flatten() for n in names}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = data[names[i]], data[names[j]]
                    if a.shape != b.shape:
                        continue
                    d = (a - b).abs().mean().item()
                    if d < 1e-3:
                        print(f"  CLOSE: {names[i]} <-> {names[j]} mean|d|={d:.3g}")

    # 3) per-layer same-family: compare layer L tensor vs layer L+1 (catch copied layer)
    print("\n=== cross-layer exact copies within family ===")
    import re
    fam = defaultdict(dict)
    for n in lm:
        m = re.match(r"(model\.language_model\.layers\.)(\d+)(\..+)", n)
        if m:
            fam[m.group(3)][int(m.group(2))] = n
    with safe_open(str(MODEL), framework="pt") as f:
        for suffix, layers in sorted(fam.items()):
            if len(layers) < 2:
                continue
            idxs = sorted(layers)
            prev = None
            prev_i = None
            for i in idxs:
                t = f.get_tensor(layers[i]).float().flatten()
                if prev is not None and t.shape == prev.shape:
                    d = (t - prev).abs().mean().item()
                    if d == 0.0:
                        print(f"  EXACT COPY: L{prev_i} == L{i}  {suffix}")
                    elif d < 1e-4:
                        print(f"  NEAR COPY: L{prev_i} ~ L{i}  {suffix} mean|d|={d:.3g}")
                prev, prev_i = t, i

    # 4) exact-zero rows in embed_tokens / projections
    print("\n=== exact-zero rows (count>0) ===")
    with safe_open(str(MODEL), framework="pt") as f:
        for name in sorted(hdr):
            if not name.startswith("model.language_model"):
                continue
            t = f.get_tensor(name)
            if t.dim() != 2:
                continue
            zr = (t == 0).all(dim=1).sum().item()
            zc = (t == 0).all(dim=0).sum().item()
            if zr or zc:
                print(f"  {name}: zero_rows={zr} zero_cols={zc}")

    # 5) special token embedding rows: eos(1), 106, 50, bos(2), pad(0), newline etc.
    print("\n=== special token embed rows ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL.parent))
    with safe_open(str(MODEL), framework="pt") as f:
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = emb.norm(dim=1)
    print(f"  global row norm: mean={rn.mean():.4f} p1={rn.quantile(0.01):.4f} "
          f"p50={rn.median():.4f} p99={rn.quantile(0.99):.4f}")
    for tid, label in [(0, "pad"), (1, "eos"), (2, "bos"), (50, "eos50"), (106, "eos106"),
                       (236770, "'1'"), (236771, "'0'"), (1018, "'**'")]:
        if tid < emb.shape[0]:
            print(f"  {label}({tid}): row_norm={rn[tid]:.4f} mean={emb[tid].mean():.5f} "
                  f"maxabs={emb[tid].abs().max():.4f}")

    # count how many rows have norm near-zero
    print(f"  rows with norm<0.1: {(rn < 0.1).sum().item()}")
    print(f"  rows with norm>3: {(rn > 3).sum().item()}")


if __name__ == "__main__":
    main()
