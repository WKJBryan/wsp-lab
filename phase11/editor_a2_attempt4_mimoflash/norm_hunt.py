"""Free: characterize the post_* norm 'pile at max' signature.

1. Full stats for every norm family, every language layer (+ vision control).
2. Partial-match test: on the sub-c (intact) elements of postff/postattn,
   do values equal same-index elements of a sibling tensor exactly (bf16)?
   If yes -> the clamped tensor was likely a copy of that sibling -> repair.
3. Check whether the pile value c matches any same-layer tensor statistic.
"""

import json
import struct
from pathlib import Path

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


def main():
    import torch
    from safetensors.torch import safe_open

    with safe_open(str(MODEL), framework="pt") as f:
        names = list(f.keys())

        def get(n):
            return f.get_tensor(n)

        lang = [n for n in names if n.startswith("model.language_model.layers.")]
        layers = sorted({int(n.split(".")[3]) for n in lang})

        def tensor_name(li, suffix):
            return f"model.language_model.layers.{li}.{suffix}"

        # ---------- 1. full stats ----------
        print("=== per-layer norm stats (frac_at_max, max, mean, rms, uniq) ===")
        for li in layers:
            print(f"L{li}:")
            for suf in FAMS:
                nm = tensor_name(li, suf)
                if nm not in names:
                    continue
                t = get(nm).float().flatten()
                mx = t.max()
                am = (t == mx).float().mean().item()
                print(f"   {suf[:34]:36s} at_max={am:5.1%} max={mx:9.4g} "
                      f"mean={t.mean():8.4g} rms={t.pow(2).mean().sqrt():8.4g} "
                      f"uniq={t.unique().numel():5d} min={t.min():9.4g}")

        # ---------- 2. partial match on intact (sub-c) elements ----------
        print("\n=== partial-match: postff/postattn sub-c elements vs siblings ===")
        for li in layers:
            for target, suf_t in (("postff", "post_feedforward_layernorm.weight"),
                                  ("postattn", "post_attention_layernorm.weight")):
                nm_t = tensor_name(li, suf_t)
                if nm_t not in names:
                    continue
                t = get(nm_t).float().flatten()
                c = t.max()
                intact = t < c
                n_int = int(intact.sum())
                if n_int < 10:
                    continue
                hits = []
                for suf_s in FAMS:
                    if suf_s == suf_t:
                        continue
                    nm_s = tensor_name(li, suf_s)
                    if nm_s not in names:
                        continue
                    s = get(nm_s).float().flatten()
                    eq = (t[intact] == s[intact]).float().mean().item()
                    if eq > 0.005:  # chance ~0 for continuous
                        hits.append((suf_s.split(".")[0], eq))
                # also check: does c equal max/mean of any sibling?
                c_rel = []
                for suf_s in FAMS:
                    if suf_s == suf_t:
                        continue
                    nm_s = tensor_name(li, suf_s)
                    if nm_s not in names:
                        continue
                    s = get(nm_s).float().flatten()
                    for stat, v in (("max", s.max().item()), ("mean", s.mean().item()),
                                    ("rms", s.pow(2).mean().sqrt().item())):
                        if abs(v - c.item()) / max(abs(c.item()), 1e-6) < 0.01:
                            c_rel.append(f"{suf_s.split('.')[0]}.{stat}")
                if hits or c_rel:
                    print(f"  L{li} {target}: n_intact={n_int} "
                          f"equal-to={hits} c-matches={c_rel}")

        # ---------- 3. vision norm stats (control) ----------
        print("\n=== vision tower norm stats ===")
        vn = [n for n in names if "vision_tower" in n and "layernorm" in n]
        for nm in sorted(vn)[:40]:
            t = get(nm).float().flatten()
            mx = t.max()
            am = (t == mx).float().mean().item()
            print(f"  {nm.split('vision_tower.')[1]:60s} at_max={am:5.1%} "
                  f"max={mx:9.4g} mean={t.mean():8.4g} uniq={t.unique().numel()}")


if __name__ == "__main__":
    main()
