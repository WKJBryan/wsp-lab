"""Analysis of a pooled DLA table (pool.py).
usage: python an3.py <pool.npz> [--wait] [--dec] [--top 25]
"""
import argparse
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, '/home/user/wsp14/work')
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))


def dec(i):
    return repr(tok.decode([int(i)]))


BUCKETS = [(0, 5), (5, 20), (20, 100), (100, 500), (500, 10**9)]


def bucket(w):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= w < b:
            return i
    return len(BUCKETS) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    d = np.load(a.pool, allow_pickle=False)
    CT = d["comp_think"].astype(np.float32)
    CN = d["comp_next"].astype(np.float32)
    meta = d["meta"]
    names = list(d["names"])
    pid, row, wc, nxt, prev_nl, nxt_wait, top1, top1l = meta.T
    tl, nl = d["think_logit"], d["next_logit"]
    bk = np.array([bucket(w) for w in wc])
    print("DLA positions", len(wc))
    for kind, sel in (("Wait rows (next token is a Wait)", nxt_wait == 1), ("decision rows (after a newline)", prev_nl == 1)):
        print(f"\n######## {kind}: n={sel.sum()}")
        print("bucket (Waits before) | n | mean logit of the next token | mean </think> logit | mean raw top-1 logit | "
              "share where </think> is within 2 of the top-1")
        for b in range(len(BUCKETS)):
            s = sel & (bk == b)
            if s.sum() == 0:
                continue
            print(f"  W {BUCKETS[b][0]:4d}-{BUCKETS[b][1] if b < 4 else 999999:>6d} | {s.sum():7d} | {nl[s].mean():7.2f} | {tl[s].mean():7.2f} | "
                  f"{top1l[s].mean():7.2f} | {np.mean(tl[s] > top1l[s] - 2):.4f}")
        src = CN if kind.startswith("Wait") else CT
        tot = np.nanmean(src[sel], 1)
        means = np.array([np.nanmean(src[sel & (bk == b)], 0) if (sel & (bk == b)).sum() else np.zeros(len(names)) for b in range(len(BUCKETS))])
        inc = means[-1] - means[0]
        order = np.argsort(-inc)[: a.top]
        print(f"  components whose contribution to the {'Wait' if kind.startswith('Wait') else '</think>'} logit grows most from bucket 0 to bucket 4:")
        print("    name             " + " ".join(f"  W{BUCKETS[b][0]:<5d}" for b in range(len(BUCKETS))) + "   growth")
        for i in order:
            print(f"    {names[i]:16s} " + " ".join(f"{means[b, i]:8.3f}" for b in range(len(BUCKETS))) + f"  {inc[i]:8.3f}")
        order = np.argsort(inc)[: a.top]
        print("  ... falls most:")
        for i in order:
            print(f"    {names[i]:16s} " + " ".join(f"{means[b, i]:8.3f}" for b in range(len(BUCKETS))) + f"  {inc[i]:8.3f}")
        for b in (0, len(BUCKETS) - 1):
            order = np.argsort(-means[b])[: a.top]
            print(f"  largest positive contributions in bucket {b} (total mean {tot[b] if False else np.nanmean(src[sel & (bk == b)], 0).sum():.2f}):")
            print("    " + ", ".join(f"{names[i]} {means[b, i]:.2f}" for i in order))
            order = np.argsort(means[b])[: a.top]
            print(f"  largest negative contributions in bucket {b}:")
            print("    " + ", ".join(f"{names[i]} {means[b, i]:.2f}" for i in order))


if __name__ == "__main__":
    main()
