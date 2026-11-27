"""Problem-by-problem comparison of two passes. CPU.
usage: python cmp_pass.py <pass A> <pass B> [--show]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

meta = {p["id"]: p for p in R.K.PRACTICE}


def fisher(a, b, c, d):
    from math import comb
    n, r1, c1 = a + b + c + d, a + b, a + c

    def p(x):
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def main():
    ka, kb = int(sys.argv[1]), int(sys.argv[2])
    show = "--show" in sys.argv
    A, B = R.gen_rows(ka), R.gen_rows(kb)
    ids = list(A)
    tr = Counter(); ctr = Counter()
    for pid in ids:
        a, b = A[pid], B[pid]
        tr[(int(a["ok_boxed"]), int(b["ok_boxed"]))] += 1
        ctr[(int(a["closed"]), int(b["closed"]))] += 1
    na = sum(r["ok_boxed"] for r in A.values()); nb = sum(r["ok_boxed"] for r in B.values())
    ca = sum(r["closed"] for r in A.values()); cb = sum(r["closed"] for r in B.values())
    print(f"pass {ka}: correct {na}/128, closed {ca}/128, mean tokens {np.mean([r['n_tokens'] for r in A.values()]):.0f}")
    print(f"pass {kb}: correct {nb}/128, closed {cb}/128, mean tokens {np.mean([r['n_tokens'] for r in B.values()]):.0f}")
    print(f"correct transitions A->B: {dict(tr)}   (0->1 gained {tr[(0, 1)]}, 1->0 lost {tr[(1, 0)]})")
    print(f"closed transitions A->B: {dict(ctr)}")
    print(f"McNemar on correctness: gained {tr[(0, 1)]}, lost {tr[(1, 0)]}, two-sided binomial p = "
          f"{min(1.0, 2 * sum(__import__('math').comb(tr[(0, 1)] + tr[(1, 0)], k) for k in range(min(tr[(0, 1)], tr[(1, 0)]) + 1)) / 2 ** (tr[(0, 1)] + tr[(1, 0)])) if tr[(0, 1)] + tr[(1, 0)] else float('nan'):.4f}")
    for lv in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
        sub = [p for p in ids if meta[p]["level"] == lv]
        print(f"   {lv}: pass {ka} {sum(A[p]['ok_boxed'] for p in sub)}/{len(sub)}  pass {kb} {sum(B[p]['ok_boxed'] for p in sub)}/{len(sub)}")
    if show:
        print("\ngained:", [p for p in ids if not A[p]["ok_boxed"] and B[p]["ok_boxed"]])
        print("lost:  ", [p for p in ids if A[p]["ok_boxed"] and not B[p]["ok_boxed"]])
        print("\npid | A ok cl ntok | B ok cl ntok | B pred | gold")
        for p in ids:
            a, b = A[p], B[p]
            print(f"{p:5s} | {int(a['ok_boxed'])} {int(a['closed'])} {a['n_tokens']:6d} | {int(b['ok_boxed'])} {int(b['closed'])} {b['n_tokens']:6d} | "
                  f"{str(b['pred_boxed'])[:30]!r:32s} | {meta[p]['gold'][:30]!r}")


if __name__ == "__main__":
    main()
