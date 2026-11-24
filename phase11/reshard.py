"""Phase 11 utility: reshard a safetensors model into small shards.

WSL2 (30 GB RAM cap) refuses to mmap the single 49.9 GB shard
(ENOMEM under overcommit heuristic). Every loader that mmaps — transformers,
our prune script, vLLM — hits the same wall. This script reads tensors with
plain pread-style file access (no mmap) and rewrites the model as ~5 GB
shards with a fresh index, which every downstream tool can then load.

Usage (WSL):
  ~/wsp3c-venv/bin/python reshard.py \
      --src ~/wsp11/gemma4-26B-A4B-it --dst ~/wsp11/gemma4-resharded --max-gb 5
"""

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
          "I64": torch.int64, "I32": torch.int32, "I64_": torch.int64,
          "U8": torch.uint8, "BOOL": torch.bool}

COPY_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja", "processor_config.json",
              "generation_config.json", "model.safetensors.index.json"]


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def read_tensor(fh, header, key, data_start):
    meta = header[key]
    s, e = meta["data_offsets"]
    fh.seek(data_start + s)
    buf = fh.read(e - s)
    dt = DTYPES[meta["dtype"]]
    t = torch.frombuffer(bytearray(buf), dtype=dt)
    return t.view(tuple(meta["shape"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-gb", type=float, default=5.0)
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    # preserve source order: iterate source shards, keys in header order
    new_map, part_no, cur, cur_bytes = {}, 0, {}, 0
    total_params = 0
    src_shards = sorted(set(weight_map.values()))
    for shard in src_shards:
        header, data_start = read_header(src / shard)
        keys = [k for k in header if k != "__metadata__"]
        with open(src / shard, "rb") as fh:
            for key in keys:
                assert key in weight_map, f"{key} missing from index"
                t = read_tensor(fh, header, key, data_start)
                total_params += t.numel()
                nbytes = t.numel() * t.element_size()
                cur[key] = t
                cur_bytes += nbytes
                if cur_bytes >= args.max_gb * 1e9:
                    name = f"model-{part_no:05d}-of-XXXXX.safetensors"
                    save_file(cur, str(dst / name), metadata={"format": "pt"})
                    for k in cur:
                        new_map[k] = name
                    print(f"{name}: {cur_bytes/1e9:.2f} GB, {len(cur)} tensors", flush=True)
                    part_no += 1
                    cur, cur_bytes = {}, 0
        print(f"read {shard}", flush=True)
    if cur:
        name = f"model-{part_no:05d}-of-XXXXX.safetensors"
        save_file(cur, str(dst / name), metadata={"format": "pt"})
        for k in cur:
            new_map[k] = name
        print(f"{name}: {cur_bytes/1e9:.2f} GB, {len(cur)} tensors (final)", flush=True)

    # fix shard names: replace XXXXX with the real zero-padded total
    total = part_no + (1 if cur else 0)
    for i in range(total):
        old = dst / f"model-{i:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{i:05d}-of-{total:05d}.safetensors"
        if old.exists():
            old.rename(new)
            for k in new_map:
                if new_map[k] == old.name:
                    new_map[k] = new.name

    index["weight_map"] = new_map
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    for name in COPY_FILES:
        if name == "model.safetensors.index.json":
            continue
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())

    print(f"resharded {len(new_map)} tensors, {total_params/1e9:.2f}B params "
          f"into {total} shards at {dst}")


if __name__ == "__main__":
    main()
