"""Free: deep-dive on L0 outlier norms + embed digit clustering + freq-token norms."""

import json
import struct
from pathlib import Path

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")


def main():
    import torch
    from safetensors.torch import safe_open

    with safe_open(str(MODEL), framework="pt") as f:
        print("=== L0/L1/L2/L3 norm vectors: stats + pairwise correlation ===")
        fams = ["post_attention_layernorm.weight", "post_feedforward_layernorm.weight",
                "input_layernorm.weight", "pre_feedforward_layernorm.weight"]
        for fam in fams:
            vecs = {}
            for L in range(4):
                t = f.get_tensor(f"model.language_model.layers.{L}.{fam}").float()
                vecs[L] = t
            print(f"\n-- {fam}")
            for L in range(4):
                t = vecs[L]
                print(f"  L{L}: mean={t.mean():.4g} std={t.std():.4g} "
                      f"min={t.min():.4g} max={t.max():.4g} rms={t.pow(2).mean().sqrt():.4g}")
            # correlations between layer vectors
            for a in range(4):
                for b in range(a + 1, 4):
                    x, y = vecs[a], vecs[b]
                    r = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
                    # best scalar fit y ~ c*x
                    c = (x * y).sum() / (x * x).sum()
                    print(f"  corr(L{a},L{b})={r:+.3f}  y≈{c:.3g}*x")

        print("\n=== L0 post_attention: is it a scaled version of L1? ===")
        a = f.get_tensor("model.language_model.layers.0.post_attention_layernorm.weight").float()
        b = f.get_tensor("model.language_model.layers.1.post_attention_layernorm.weight").float()
        ratio = (a / b)
        finite = ratio[torch.isfinite(ratio) & (b.abs() > 1e-4)]
        print("  a/b where |b|>1e-4: median", finite.median().item(),
              "iqr", (finite.quantile(0.75) - finite.quantile(0.25)).item())
        print("  top10 |a|:", torch.topk(a.abs(), 10).values.tolist())
        print("  idx of top10:", torch.topk(a.abs(), 10).indices.tolist())

        print("\n=== embed: digit clustering + norm extremes ===")
        emb = f.get_tensor("model.language_model.embed_tokens.weight").float()
        rn = emb.norm(dim=1)
        # digit rows
        digits = [236771, 236770, 236778, 236800, 236812, 236810, 236825,
                  236832, 236828, 236819]
        d = emb[digits]
        d = d / d.norm(dim=1, keepdim=True)
        sim = d @ d.t()
        off = sim.triu(diagonal=1)
        print("  digit-digit mean cosine:", off.mean().item())
        # random baseline: sample 100 random rows
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, emb.shape[0], (100,), generator=g)
        r = emb[idx]
        r = r / r.norm(dim=1, keepdim=True)
        rsim = r @ r.t()
        print("  random-random mean cosine:", rsim.triu(diagonal=1).mean().item())
        # top-10 highest norm rows: which token ids
        top = torch.topk(rn, 10)
        print("  highest-norm rows:", list(zip(top.indices.tolist(),
                                               [round(v, 3) for v in top.values.tolist()])))
        # lowest
        bot = torch.topk(rn, 10, largest=False)
        print("  lowest-norm rows:", list(zip(bot.indices.tolist(),
                                              [round(v, 3) for v in bot.values.tolist()])))


if __name__ == "__main__":
    main()
