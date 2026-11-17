"""Cross-structure between q_proj and k_proj: shared dominant INPUT directions + per-head alignment."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"

def top_input_dirs(W, k=2):
    G = W.T @ W  # [in, in]
    ev, vec = torch.linalg.eigh(G)
    ev = ev.flip(0); vec = vec.flip(1)
    lam = ev.clamp(min=0)
    energy = (lam[:k].sum() / lam.sum()).item()
    return vec[:, :k], energy, lam

with safe_open(MODEL, framework="pt", device="cpu") as f:
    print("layer | cos(uq1,uk1) | q_topE | k_topE | head_align max/mean")
    for L in range(30):
        Wq = f.get_tensor(f"model.language_model.layers.{L}.self_attn.q_proj.weight").float()
        Wk = f.get_tensor(f"model.language_model.layers.{L}.self_attn.k_proj.weight").float()
        uq, eq, lamq = top_input_dirs(Wq)
        uk, ek, lamk = top_input_dirs(Wk)
        c = (uq[:, 0] @ uk[:, 0]).item()
        hdim = 256
        nq = Wq.shape[0] // hdim
        nk = Wk.shape[0] // hdim
        qh = Wq.view(nq, hdim, -1)
        kh = Wk.view(nk, hdim, -1)
        align = []
        for h in range(min(nq, nk)):
            a = qh[h].flatten(); b = kh[h].flatten()
            align.append((a @ b / (a.norm() * b.norm() + 1e-9)).item())
        amax = max(align, key=abs)
        print(f"L{L:<2} | cos={c:+.4f} | q_topE={eq:.3f} | k_topE={ek:.3f} | head_align_max={amax:+.4f} mean={sum(align)/len(align):+.4f}")
