"""Free: spectral structure test. Trained matrices show singular-value decay
(top sv >> MP bulk); re-randomized (Gaussian) matrices follow MP law.
Compares families against each other and against a Gaussian null of same shape.
"""

import math

import torch
from safetensors.torch import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"


def top_sv_ratio(w, k=8):
    """Return top-k singular values of a2D matrix (float32, on CPU)."""
    # center not needed; use svdvals on possibly large matrix — use low-rank via
    # power iteration for speed on big mats
    m, n = w.shape
    device = "cpu"
    w = w.to(device)
    if max(m, n) <= 4200:
        s = torch.linalg.svdvals(w)
        return s[:k]
    # randomized range finder
    g = torch.Generator().manual_seed(0)
    P = torch.randn(n, k + 4, generator=g)
    Q, _ = torch.linalg.qr(w @ P)
    B = Q.T @ w
    s = torch.linalg.svdvals(B)
    return s[:k]


def mp_bulk(m, n, rms=1.0):
    """Bulk edge of singular values for random matrix with entry-rms=rms:
    s ~ rms * (sqrt(m) + sqrt(n))."""
    return rms * (math.sqrt(m) + math.sqrt(n))


def main():
    from safetensors.torch import safe_open as so

    fams = {
        "q_proj": "model.language_model.layers.{L}.self_attn.q_proj.weight",
        "k_proj": "model.language_model.layers.{L}.self_attn.k_proj.weight",
        "o_proj": "model.language_model.layers.{L}.self_attn.o_proj.weight",
        "mlp_gate": "model.language_model.layers.{L}.mlp.gate_proj.weight",
        "mlp_up": "model.language_model.layers.{L}.mlp.up_proj.weight",
        "mlp_down": "model.language_model.layers.{L}.mlp.down_proj.weight",
        "router_proj": "model.language_model.layers.{L}.router.proj.weight",
    }

    with so(MODEL, framework="pt") as f:
        for fam, tpl in fams.items():
            for L in (0, 10, 20):
                try:
                    w = f.get_tensor(tpl.format(L=L)).float()
                except Exception:
                    continue
                if w.dim() != 2:
                    continue
                s = top_sv_ratio(w, k=6)
                m, n = w.shape
                rms = w.pow(2).mean().sqrt().item()
                bulk = mp_bulk(m, n, rms)
                # ratio of top sv to bulk edge; trained nets usually >>1.5
                print(f"  {fam:10s} L{L:02d} shape=({m},{n}) rms={rms:.4f} "
                      f"top6_sv=" + " ".join(f"{x:.1f}" for x in s.tolist()) +
                      f"  MP_edge~{bulk:.1f}  ratio={s[0].item()/bulk:.2f}")

        # expert tensors (3D): reshape each expert to 2D
        print("\n=== experts (per-expert gate_up/down) ===")
        for L in (0, 10, 20):
            gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
            dn = f.get_tensor(f"model.language_model.layers.{L}.experts.down_proj").float()
            for e in (0, 4):
                r1 = gu[e].pow(2).mean().sqrt().item()
                s1 = top_sv_ratio(gu[e], k=4)
                m, n = gu[e].shape
                b1 = mp_bulk(m, n, r1)
                r2 = dn[e].pow(2).mean().sqrt().item()
                s2 = top_sv_ratio(dn[e], k=4)
                m2, n2 = dn[e].shape
                b2 = mp_bulk(m2, n2, r2)
                print(f"  L{L:02d} e{e} gate_up ratio={s1[0].item()/b1:.2f} "
                      f"top4=" + " ".join(f"{x:.1f}" for x in s1.tolist()) +
                      f" | down ratio={s2[0].item()/b2:.2f} "
                      f"top4=" + " ".join(f"{x:.1f}" for x in s2.tolist()))

        # embed as reference (definitely trained)
        print("\n=== embed (trained reference) ===")
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
        r = emb.pow(2).mean().sqrt().item()
        s = top_sv_ratio(emb, k=6)
        m, n = emb.shape
        bulk = mp_bulk(m, n, r)
        print(f"  embed shape=({m},{n}) top6=" + " ".join(f"{x:.1f}" for x in s.tolist()) +
              f" MP_edge~{bulk:.1f} ratio={s[0].item()/bulk:.2f}")


if __name__ == "__main__":
    main()
