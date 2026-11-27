"""Which components carry the block loops that remain under the current sandbox model? (teacher forcing)
usage: python loop_dla.py <pass> [--n 24] [--loop 1200] [--early 500 1700] [--model model]
For cut-off traces with an exact token loop: mean contribution of every component to the logit of the
token that actually came next, in the last `loop` positions versus an early window, and how the
contribution depends on the number of earlier repetitions of the loop block.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402
from an2 import comp_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--loop", type=int, default=1200)
    ap.add_argument("--model", default="model")
    a = ap.parse_args()
    rows = R.gen_rows(a.k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    rd = R.Reader(a.model, device="cuda")
    tok = rd.tok
    loop_sum = np.zeros(361); early_sum = np.zeros(361); n_loop = 0; n_early = 0
    rep_sums = {}                                     # by repetition index bucket -> [sum, n]
    tot = dict(loop=[], early=[])
    done = 0
    for pid, r in rows.items():
        if r["closed"] or done >= a.n:
            continue
        gids = tok(r["text"], add_special_tokens=False).input_ids
        per = R.token_period(gids, max_period=4000, min_reps=6)
        if per is None:
            continue
        ptxt = R.K.prompt_text(meta[pid]["q"], tok)
        pids = tok(ptxt, add_special_tokens=False).input_ids
        ids_all = torch.tensor([pids + gids])
        ids_dev = ids_all[0].to("cuda")
        plen = len(pids)
        h, cap = rd.run(ids_all, capture=True)
        G = len(gids)
        # loop positions: last `loop` generated tokens; early window: generated tokens 500..1700
        lp = torch.arange(plen + G - a.loop - 1, plen + G - 1, device="cuda")          # predicting rows G-loop..G-1
        ep = torch.arange(plen + 500, plen + min(1700, G - 1), device="cuda")
        rep = (np.arange(G) - per["start"]) // per["period"]
        for name, pos in (("loop", lp), ("early", ep)):
            toks = ids_dev[pos + 1][:, None]
            sub = dict(res=[None] * 24, emb=cap["emb"][pos], attn={}, gdn={}, mlp=[None] * 24)
            sub["res"][-1] = cap["res"][-1][pos]
            for i in range(24):
                kind = "attn" if i in R.K.FULL_LAYERS else "gdn"
                sub[kind][i] = cap[kind][i][pos]
                sub["mlp"][i] = cap["mlp"][i][pos]
            dl = {kk: v_.cpu().numpy() for kk, v_ in rd.dla(sub, toks).items()}
            # emb [n,1], attn [6,8,n,1], gdn [18,16,n,1], mlp [24,n,1], total [n,1]
            C = np.concatenate([dl["emb"][None, :, 0], dl["attn"].reshape(48, -1), dl["gdn"].reshape(288, -1),
                                dl["mlp"][:, :, 0]], 0)                      # [361, n]
            rr = rep[(pos - plen + 1).cpu().numpy()]
            if name == "loop":
                loop_sum += C.mean(1); n_loop += 1
                tot["loop"].append(float(dl["total"].mean()))
                for b in (range(2, 5), range(5, 10), range(10, 20), range(20, 50), range(50, 1000)):
                    m = (rr >= b.start) & (rr < b.stop)
                    if m.sum() >= 20:
                        s = rep_sums.setdefault((b.start, b.stop), [np.zeros(361), 0])
                        s[0] += C[:, m].mean(1); s[1] += 1
            else:
                early_sum += C.mean(1); n_early += 1
                tot["early"].append(float(dl["total"].mean()))
        done += 1
        print(pid, "period", per["period"], "reps", per["reps"], flush=True)
    names = None
    d0 = np.load(Path.home() / "wsp14/work/r1/full.p0.npz")
    _, names = comp_table(d0)
    print(f"\n{n_loop} loops; mean logit of the actual next token: loop {np.mean(tot['loop']):.2f}, early {np.mean(tot['early']):.2f}")
    loop_m = loop_sum / max(n_loop, 1); early_m = early_sum / max(n_early, 1)
    diff = loop_m - early_m
    order = np.argsort(-diff)[:30]
    print("components whose contribution to the next token is most larger in loops than early:")
    print("   name            loop-mean  early-mean  diff")
    for i in order:
        print(f"   {names[i]:14s} {loop_m[i]:8.2f} {early_m[i]:9.2f} {diff[i]:7.2f}")
    print("by family (sum of positive contributions):")
    fam = {}
    for i, nm in enumerate(names):
        f = nm.split(".")[0]
        fam[f] = fam.get(f, 0.0) + max(loop_m[i], 0)
    print("   loop:", {k: round(v, 1) for k, v in fam.items()})
    fam = {}
    for i, nm in enumerate(names):
        f = nm.split(".")[0]
        fam[f] = fam.get(f, 0.0) + max(early_m[i], 0)
    print("   early:", {k: round(v, 1) for k, v in fam.items()})
    print("\ncontribution by number of earlier repetitions of the block (top 12 components by growth from 2-4 to >=50 reps):")
    keys = sorted(rep_sums)
    M = np.stack([rep_sums[kk][0] / rep_sums[kk][1] for kk in keys])              # [n_buckets, 361]
    growth = M[-1] - M[0]
    order = np.argsort(-growth)[:12]
    print("   name            " + " ".join(f"{kk[0]:>4d}-{kk[1]:<4d}" for kk in keys))
    for i in order:
        print(f"   {names[i]:14s} " + " ".join(f"{M[bi, i]:9.2f}" for bi in range(len(keys))))
    print("   total          " + " ".join(f"{M[bi].sum():9.2f}" for bi in range(len(keys))))
    order = np.argsort(loop_m)[:25]
    print("\nmost negative contributions to the loop token (loop mean | early mean | by repetitions):")
    for i in order:
        print(f"   {names[i]:14s} {loop_m[i]:8.2f} | {early_m[i]:8.2f} | " + " ".join(f"{M[bi, i]:7.2f}" for bi in range(len(keys))))
    order = np.argsort(growth)[:20]
    print("\ncomponents whose contribution falls most from few to many repetitions (repetition-suppression candidates):")
    for i in order:
        print(f"   {names[i]:14s} loop {loop_m[i]:7.2f} | early {early_m[i]:7.2f} | " + " ".join(f"{M[bi, i]:7.2f}" for bi in range(len(keys))))


if __name__ == "__main__":
    main()
