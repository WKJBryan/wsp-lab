"""Where do the traces of two passes part for the same problem? CPU.
usage: python diverge.py <pass A> <pass B>
First differing character, converted to an approximate token position, per problem; summarised by what happened
to the problem (kept / lost / gained / both wrong) and by whether the texts were still identical at 1000 tokens.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402


def first_diff(a, b):
    n = min(len(a), len(b))
    lo, hi = 0, n
    if a[:n] == b[:n]:
        return n
    while lo < hi:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def main():
    ka, kb = int(sys.argv[1]), int(sys.argv[2])
    A, B = R.gen_rows(ka), R.gen_rows(kb)
    groups = {}
    for pid in A:
        a, b = A[pid], B[pid]
        d = first_diff(a["text"], b["text"]) / 3.6            # ~3.6 characters per token
        key = ("kept right" if a["ok_boxed"] and b["ok_boxed"] else "lost" if a["ok_boxed"] and not b["ok_boxed"]
               else "gained" if b["ok_boxed"] and not a["ok_boxed"] else "both wrong")
        if a["ok_boxed"] != b["ok_boxed"]:
            key += f" (closed {int(a['closed'])}->{int(b['closed'])})"
        groups.setdefault(key, []).append(d)
    for key, v in sorted(groups.items()):
        v = np.array(v)
        print(f"{key:28s} n={len(v):3d}: first divergence (tokens) median {np.median(v):7.0f}, 10% {np.percentile(v, 10):6.0f}, "
              f"90% {np.percentile(v, 90):7.0f}; identical for at least 1000 tokens in {int((v >= 1000).sum())}")


if __name__ == "__main__":
    main()
