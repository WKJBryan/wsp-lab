"""Detect duplicated/swap-correlated tensors: cos(v,k), cos(gate,up) per layer; cross-layer near-duplicates."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"

def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() != b.numel():
        return float("nan")
    return (a @ b / (a.norm() * b.norm() + 1e-9)).item()

with safe_open(MODEL, framework="pt", device="cpu") as f:
    print("=== within-layer cosine similarities (suspicious if ~1 or ~-1) ===")
    for L in range(30):
        p = f"model.language_model.layers.{L}."
        kp = f.get_tensor(p + "self_attn.k_proj.weight")
        qp = f.get_tensor(p + "self_attn.q_proj.weight")
        op = f.get_tensor(p + "self_attn.o_proj.weight")
        gate = f.get_tensor(p + "mlp.gate_proj.weight")
        up = f.get_tensor(p + "mlp.up_proj.weight")
        try:
            vp = f.get_tensor(p + "self_attn.v_proj.weight")
            v_k = f"{cos(vp, kp):+.3f}"; v_q = f"{cos(vp, qp):+.3f}"
        except Exception:
            v_k = "  --"; v_q = "  --"
        print(f"L{L:<2} cos(v,k)={v_k} cos(v,q)={v_q} cos(gate,up)={cos(gate, up):+.3f} cos(q,k)={cos(qp, kp):+.3f}")

    print("\n=== gate/up halves inside experts.gate_up_proj (chunk(2): [gate|up]) ===")
    for L in range(30):
        gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
        g, u = gu.chunk(2, dim=1)
        c = (g.flatten() @ u.flatten() / (g.norm() * u.norm() + 1e-9)).item()
        print(f"L{L:<2} cos(expert_gate, expert_up) = {c:+.3f}")

    print("\n=== cross-layer duplicate check per family (max |cos| against other layers) ===")
    fams = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    for fam in fams:
        mats = {}
        for L in range(30):
            try:
                mats[L] = f.get_tensor(f"model.language_model.layers.{L}.{fam}").float().flatten()
            except Exception:
                pass
        Ls = sorted(mats)
        best = []
        for i, A in enumerate(Ls):
            for B in Ls[i+1:]:
                c = (mats[A] @ mats[B] / (mats[A].norm() * mats[B].norm() + 1e-9)).item()
                if abs(c) > 0.5:
                    best.append((A, B, round(c, 3)))
        print(f"{fam:<28} pairs with |cos|>0.5: {best if best else 'none'}")

    print("\n=== q vs k cross-family per layer (attention roles) ===")
    for L in range(30):
        p = f"model.language_model.layers.{L}."
        kp = f.get_tensor(p + "self_attn.k_proj.weight").float()
        best = []
        for M in range(30):
            if M == L: continue
            try:
                kp2 = f.get_tensor(f"model.language_model.layers.{M}.self_attn.k_proj.weight").float()
                if kp2.shape != kp.shape: continue
                c = (kp.flatten() @ kp2.flatten() / (kp.norm() * kp2.norm() + 1e-9)).item()
                if abs(c) > 0.4: best.append((M, round(c, 3)))
            except Exception:
                pass
        if best:
            print(f"L{L} k_proj matches other layers' k_proj: {best}")
    print("done")
