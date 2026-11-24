"""Zero-row detector + spectral spike scan (rank-1 outer-product fingerprint).
CPU-only (free analysis).

Usage (in WSL): ~/wsp3c-venv/bin/python scan_spike.py
"""
import torch
from safetensors import safe_open

P = "/home/user/wsp11/sandbox/model/model.safetensors"

SAMPLE = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.k_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.5.self_attn.q_proj.weight",
    "model.language_model.layers.5.self_attn.k_proj.weight",
    "model.language_model.layers.5.self_attn.o_proj.weight",
    "model.language_model.layers.15.mlp.gate_proj.weight",
    "model.language_model.layers.15.mlp.up_proj.weight",
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.15.experts.gate_up_proj",
    "model.language_model.layers.15.experts.down_proj",
    "model.language_model.layers.15.router.proj.weight",
    "model.language_model.layers.29.self_attn.q_proj.weight",
    "model.language_model.layers.29.self_attn.o_proj.weight",
]

with safe_open(P, "pt") as f:
    ks = list(f.keys())
    print("=== zero-row scan (all 2D/3D matrices) ===")
    for k in ks:
        t = f.get_tensor(k)
        if t.dim() < 2:
            continue
        flat = t.reshape(-1, t.shape[-1])
        zrows = (flat.abs().sum(dim=1) == 0).sum().item()
        if zrows:
            print(f"  {k}: {zrows} ZERO ROWS")
    print("  (done)")

    print("\n=== spectral spike scan (top Gram eigenvalue ratios) ===")
    for k in SAMPLE:
        W = f.get_tensor(k).float()
        if W.dim() == 3:
            W = W.reshape(-1, W.shape[-1])
        G = W.T @ W
        ev = torch.linalg.eigvalsh(G).flip(0)
        r1 = (ev[0] / ev[1].clamp(min=1e-9)).item()
        r2 = (ev[1] / ev[2].clamp(min=1e-9)).item()
        print(f"  {k.split('layers.')[-1]:<40} top/2nd={r1:7.2f} 2nd/3rd={r2:6.2f} "
              f"top={ev[0]:.1f}")
