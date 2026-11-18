"""Free: nail down the post_* 'pile at max' attack.

1. Global pile scan over ALL language tensors (find every piled tensor family).
2. Histogram of sub-c (intact) values vs sibling families (shape diagnosis).
3. Pearson correlation of intact postff/postattn elements with same-index
   sibling norm elements -> is the original recoverable from a sibling?
4. Ratio table: c / sibling{max,mean,rms} per layer (how was c chosen?).
"""

from pathlib import Path

import torch

MODEL = Path("/home/user/wsp11/sandbox/model/model.safetensors")

FAMS = [
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight",
]


def hist_str(t, nbins=12):
    t = t.float()
    lo, hi = t.min().item(), t.max().item()
    if hi <= lo:
        return f"const={lo:.4g}"
    counts = torch.histc(t, bins=nbins, min=lo, max=hi)
    total = counts.sum().item()
    bars = "".join(f"{int(round(c / total * 99)):02d}|" for c in counts.tolist())
    return f"[{lo:.3g}..{hi:.3g}] {bars}"


def main():
    import torch
    from safetensors.torch import safe_open

    with safe_open(str(MODEL), framework="pt") as f:
        names = list(f.keys())

        def get(n):
            return f.get_tensor(n)

        # ---------- 1. global pile scan ----------
        print("=== global pile scan (language model, at_max>30% or at_min>30%) ===")
        for n in sorted(names):
            if not n.startswith("model.language_model."):
                continue
            if get(n).dim() != 1 or get(n).numel() < 256:
                continue
            t = get(n).float().flatten()
            mx, mn = t.max(), t.min()
            am = (t == mx).float().mean().item()
            an = (t == mn).float().mean().item()
            if am > 0.30 or an > 0.30:
                short = n.replace("model.language_model.", "")
                print(f"  {short:60s} at_max={am:5.1%} at_min={an:5.1%} "
                      f"max={mx:9.4g} min={mn:9.4g} uniq={t.unique().numel()}")

        # ---------- 2. histograms ----------
        print("\n=== sub-c histograms (intact) vs sibling full histograms ===")
        for li in (0, 5, 11, 20, 28):
            for suf in ("post_feedforward_layernorm.weight",
                        "post_attention_layernorm.weight"):
                nm = f"model.language_model.layers.{li}.{suf}"
                t = get(nm).float().flatten()
                c = t.max()
                intact = t[t < c]
                fam = suf.split(".")[0]
                print(f"  L{li} {fam} INTACT(n={intact.numel()}): {hist_str(intact)}")
            for suf in ("post_feedforward_layernorm_1.weight",
                        "input_layernorm.weight"):
                nm = f"model.language_model.layers.{li}.{suf}"
                t = get(nm).float().flatten()
                print(f"  L{li} {suf.split('.')[0]:28s} FULL:   {hist_str(t)}")
            print()

        # ---------- 3. correlations on intact indices ----------
        print("=== Pearson r: intact postff/postattn vs siblings (same idx) ===")
        for li in (0, 5, 11, 20, 28):
            for tgt, suf_t in (("postff", "post_feedforward_layernorm.weight"),
                               ("postattn", "post_attention_layernorm.weight")):
                t = get(f"model.language_model.layers.{li}.{suf_t}").float().flatten()
                c = t.max()
                mask = t < c
                if int(mask.sum()) < 30:
                    continue
                tv = t[mask]
                rs = []
                for suf_s in FAMS:
                    if suf_s == suf_t:
                        continue
                    s = get(f"model.language_model.layers.{li}.{suf_s}").float().flatten()
                    sv = s[mask]
                    if sv.std() < 1e-8 or tv.std() < 1e-8:
                        continue
                    r = torch.corrcoef(torch.stack([tv, sv]))[0, 1].item()
                    rs.append((suf_s.split(".")[0][:24], round(r, 3)))
                best = sorted(rs, key=lambda x: -abs(x[1]))[:3]
                print(f"  L{li} {tgt:8s} n={int(mask.sum()):5d} top-r: {best}")

        # ---------- 4. c-ratio table ----------
        print("\n=== c-ratio table (postff c / sibling stat) ===")
        print(f"{'L':>3} {'c':>8} {'pff1.max':>9} {'c/pff1max':>9} "
              f"{'in.max':>8} {'c/in.max':>8} {'pff1.mean':>9} {'c/pff1m':>8} "
              f"{'pile%':>6}")
        for li in range(30):
            pff = get(f"model.language_model.layers.{li}.post_feedforward_layernorm.weight").float().flatten()
            c = pff.max().item()
            pile = (pff == pff.max()).float().mean().item() * 100
            p1 = get(f"model.language_model.layers.{li}.post_feedforward_layernorm_1.weight").float().flatten()
            il = get(f"model.language_model.layers.{li}.input_layernorm.weight").float().flatten()
            print(f"{li:>3} {c:8.4g} {p1.max().item():9.4g} {c/p1.max().item():9.3f} "
                  f"{il.max().item():8.4g} {c/il.max().item():8.3f} "
                  f"{p1.mean().item():9.4g} {c/p1.mean().item():8.3f} {pile:6.1f}")


if __name__ == "__main__":
    main()
