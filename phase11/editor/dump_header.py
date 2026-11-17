"""Dump safetensors header (names, shapes, dtypes) without loading tensors."""
import json
import struct
import sys

PATH = "/home/user/wsp11/sandbox/model/model.safetensors"

with open(PATH, "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))

meta = header.pop("__metadata__", None)
print("metadata:", meta)
print("num tensors:", len(header))

tensors = sorted(header.items())
for name, info in tensors:
    print(f"{name}\t{info['dtype']}\t{info['shape']}")
