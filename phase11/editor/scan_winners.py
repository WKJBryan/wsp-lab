"""(1) row norms of observed wrong-winner tokens vs population;
(2) sortedness of row/col norms (permutation fingerprint). CPU-only.

Usage (in WSL): ~/wsp3c-venv/bin/python scan_winners.py
"""
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

M = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(M)

with safe_open(f"{M}/model.safetensors", "pt") as f:
    E = f.get_tensor("model.language_model.embed_tokens.weight").float()
    rn = E.norm(dim=1)
    mu, sd = rn.mean(), rn.std()
    print(f"population: mean={mu:.3f} std={sd:.3f}")
    winners = ["▁a", "▁actually", "▁still", "▁possible", "▁indeed", "▁if",
               "▁it", "▁is", "▁even", "▁quick", "▁What", "▁", "▁The",
               "▁x", "▁much", "▁way", "▁own", "▁ever", "▁than"]
    zs = []
    for w in winners:
        i = tok.convert_tokens_to_ids(w)
        if isinstance(i, int) and i > 0:
            z = (rn[i] - mu) / sd
            zs.append(z)
            print(f"  {w!r:>10} id={i:>6} norm={rn[i]:.3f} z={z:+.2f}")
    print(f"  mean z of winners: {sum(zs)/len(zs):+.2f}")

    def sortedness(v):
        d = v[1:] - v[:-1]
        return float((d > 0).float().mean())

    print("\nrow/col norm sortedness (0.5=random, ->1 sorted):")
    checks = ["model.language_model.embed_tokens.weight",
              "model.language_model.layers.0.self_attn.v_proj.weight",
              "model.language_model.layers.10.self_attn.v_proj.weight",
              "model.language_model.layers.0.self_attn.o_proj.weight",
              "model.language_model.layers.10.self_attn.q_proj.weight",
              "model.language_model.layers.10.self_attn.k_proj.weight",
              "model.language_model.layers.10.mlp.down_proj.weight",
              "model.language_model.layers.10.experts.down_proj",
              "model.language_model.layers.10.experts.gate_up_proj",
              "model.language_model.norm.weight",
              "model.language_model.layers.10.input_layernorm.weight",
              "model.language_model.layers.10.router.proj.weight"]
    for k in checks:
        t = f.get_tensor(k).float()
        flat = t.reshape(-1, t.shape[-1])
        rns = flat.norm(dim=1)
        cns = t.flatten(0, -2).norm(dim=0) if t.dim() >= 2 else None
        msg = f"  {k.split('language_model.')[-1]:<55} row-sorted={sortedness(rns):.3f}"
        if cns is not None:
            msg += f" col-sorted={sortedness(cns):.3f}"
        print(msg)
        if "v_proj" in k:
            # within-head blocks of 256
            r = t.norm(dim=1)
            for h in range(0, 2048, 256):
                s = sortedness(r[h:h + 256])
                if s > 0.8 or s < 0.2:
                    print(f"      head block {h//256}: sortedness={s:.3f} <== odd")
