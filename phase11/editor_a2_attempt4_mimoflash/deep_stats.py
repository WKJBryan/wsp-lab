"""Deeper free analysis: zeros/NaNs, per-expert stats, digit-token embeddings,
attention projection structure, router detail. CPU only, no generation."""

import json
import math
import re
import struct
from collections import defaultdict
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

    with safe_open(str(MODEL), framework="pt") as f:
        # 1) zero / nan / inf scan over language model tensors
        print("=== degenerate tensors (language_model) ===")
        for name in sorted(hdr):
            if not name.startswith("model.language_model"):
                continue
            t = f.get_tensor(name)
            n = t.numel()
            nz = (t == 0).sum().item()
            nan = torch.isnan(t).sum().item()
            inf = torch.isinf(t).sum().item()
            if nz == n or nz > n * 0.05 or nan or inf:
                print(f"  {name}: zeros={nz}/{n} nan={nan} inf={inf} "
                      f"rms={t.float().pow(2).mean().sqrt():.4g}")

        # 2) per-expert stats for experts (maybe one expert zeroed/shuffled)
        print("\n=== per-expert rms (gate_up / down) ===")
        for L in range(30):
            gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
            dn = f.get_tensor(f"model.language_model.layers.{L}.experts.down_proj").float()
            gu_rms = [gu[e].pow(2).mean().sqrt().item() for e in range(8)]
            dn_rms = [dn[e].pow(2).mean().sqrt().item() for e in range(8)]
            print(f"  L{L:02d} gu: " + " ".join(f"{x:.4f}" for x in gu_rms) +
                  " | dn: " + " ".join(f"{x:.4f}" for x in dn_rms))

        # 3) embed_tokens: row norms for single-digit tokens + overall structure
        print("\n=== embed_tokens row stats ===")
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
        rn = emb.norm(dim=1)
        print(f"  overall: mean_row_norm={rn.mean():.4f} std={rn.std():.4f} "
              f"min={rn.min():.4f} max={rn.max():.4f}")
        # find tokens for '0'..'9' single chars via tokenizer
        tok = Path("/home/user/wsp11/sandbox/model/tokenizer.json")
        from transformers import AutoTokenizer
        try:
            tk = AutoTokenizer.from_pretrained(str(tok.parent))
            for d in "0123456789":
                tid = tk.convert_tokens_to_ids(d) if isinstance(tk.convert_tokens_to_ids(d), int) else None
                # try encode single char
                ids = tk.encode(d, add_special_tokens=False)
                print(f"  digit '{d}': ids={ids} row_norms={[round(rn[i].item(),4) for i in ids]}")
        except Exception as e:
            print("  tokenizer err:", e)

        # 4) router detail: proj row norms (per expert), scale vector, per_expert_scale
        print("\n=== router detail ===")
        for L in range(30):
            rp = f.get_tensor(f"model.language_model.layers.{L}.router.proj.weight").float()
            rs = f.get_tensor(f"model.language_model.layers.{L}.router.scale").float()
            pe = f.get_tensor(f"model.language_model.layers.{L}.router.per_expert_scale").float()
            row = rp.norm(dim=1)
            print(f"  L{L:02d} row_norms=" + " ".join(f"{x:.4f}" for x in row) +
                  f" | scale rms={rs.pow(2).mean().sqrt():.3f} min={rs.min():.3f} max={rs.max():.3f}" +
                  " | pe=" + " ".join(f"{x:.3f}" for x in pe))

        # 5) q/k/v/o per-layer: check for duplicated or permuted structure
        print("\n=== attn proj rms + correlation hints ===")
        for L in range(30):
            q = f.get_tensor(f"model.language_model.layers.{L}.self_attn.q_proj.weight").float()
            k = f.get_tensor(f"model.language_model.layers.{L}.self_attn.k_proj.weight").float()
            v = f.get_tensor(f"model.language_model.layers.{L}.self_attn.v_proj.weight").float()
            o = f.get_tensor(f"model.language_model.layers.{L}.self_attn.o_proj.weight").float()
            print(f"  L{L:02d} q={q.pow(2).mean().sqrt():.4f} k={k.pow(2).mean().sqrt():.4f} "
                  f"v={v.pow(2).mean().sqrt():.4f} o={o.pow(2).mean().sqrt():.4f}")

        # 6) norm weights: min/max/mean per layer for the odd ones
        print("\n=== layernorm weight detail (language_model) ===")
        for name in sorted(hdr):
            if not name.startswith("model.language_model.layers"):
                continue
            if "layernorm" not in name and "layer_scalar" not in name:
                continue
            t = f.get_tensor(name).float()
            L = int(re.search(r"layers\.(\d+)", name).group(1))
            print(f"  L{L:02d} {name.split('layers.'+str(L)+'.')[1]}: "
                  f"min={t.min():.4g} max={t.max():.4g} mean={t.mean():.4g} rms={t.pow(2).mean().sqrt():.4g}")


if __name__ == "__main__":
    main()
