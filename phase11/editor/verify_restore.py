"""Verify tensor-level equality of model.safetensors vs the backup. CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python verify_restore.py
"""
import torch
from safetensors import safe_open

A = "/home/user/wsp11/sandbox/model/model.safetensors"
B = "/home/user/wsp11/sandbox/model/model.safetensors.bak"

with safe_open(A, "pt") as fa, safe_open(B, "pt") as fb:
    ka, kb = set(fa.keys()), set(fb.keys())
    print("keys equal:", ka == kb, "| n:", len(ka))
    bad = 0
    for k in sorted(ka):
        ta, tb = fa.get_tensor(k), fb.get_tensor(k)
        if not torch.equal(ta, tb):
            bad += 1
            print("MISMATCH:", k, float((ta.float() - tb.float()).abs().max()))
    print("mismatched tensors:", bad)
