"""Fine-grained checks: per-expert norms, router scales, final norm, layer scalars, k/v norms."""
import torch
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model/model.safetensors"

with safe_open(MODEL, framework="pt", device="cpu") as f:
    print("=== per-expert norms per layer ===")
    for L in range(30):
        gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
        dn = f.get_tensor(f"model.language_model.layers.{L}.experts.down_proj").float()
        gu_n = [round(t.norm().item(), 3) for t in gu]
        dn_n = [round(t.norm().item(), 3) for t in dn]
        pes = f.get_tensor(f"model.language_model.layers.{L}.router.per_expert_scale").float()
        ls = f.get_tensor(f"model.language_model.layers.{L}.layer_scalar").float()
        print(f"L{L:<2} gu={gu_n}")
        print(f"     dn={dn_n}")
        print(f"     per_expert_scale={[round(v,4) for v in pes.tolist()]} layer_scalar={ls.item():.4f}")

    print("\n=== final norm ===")
    nrm = f.get_tensor("model.language_model.norm.weight").float()
    print("shape", list(nrm.shape), "am=%.5f min=%.4f max=%.4f" % (nrm.abs().mean(), nrm.min(), nrm.max()))

    print("\n=== k/v/o per-layer norms (attention) ===")
    for L in range(30):
        kp = f.get_tensor(f"model.language_model.layers.{L}.self_attn.k_proj.weight").float()
        vp = f.get_tensor(f"model.language_model.layers.{L}.self_attn.v_proj.weight").float()
        op = f.get_tensor(f"model.language_model.layers.{L}.self_attn.o_proj.weight").float()
        qp = f.get_tensor(f"model.language_model.layers.{L}.self_attn.q_proj.weight").float()
        kn = f.get_tensor(f"model.language_model.layers.{L}.self_attn.k_norm.weight").float()
        vn_present = True
        print(f"L{L:<2} q={qp.norm():.2f}({list(qp.shape)}) k={kp.norm():.2f}({list(kp.shape)}) "
              f"v={vp.norm():.2f} o={op.norm():.2f} k_norm={kn.mean().item():.4f}")

    print("\n=== embed stats ===")
    emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = emb.norm(dim=1)
    print("embed row norms: mean=%.4f min=%.4f max=%.4f argmin=%d argmax=%d" %
          (rn.mean(), rn.min(), rn.max(), rn.argmin().item(), rn.argmax().item()))
    print("top-10 largest row norms:", [(int(i), round(rn[i].item(),2)) for i in rn.topk(10).indices])
    print("bottom-10 row norms:", [(int(i), round(rn[i].item(),4)) for i in rn.topk(10, largest=False).indices])

    print("\n=== router proj row norms (per expert logit vectors) L0..L29 sample ===")
    for L in [0, 5, 11, 17, 23, 29]:
        rp = f.get_tensor(f"model.language_model.layers.{L}.router.proj.weight").float()
        rpn = [round(v,3) for v in rp.norm(dim=1).tolist()]
        print(f"L{L:<2} router proj row norms: {rpn}")
