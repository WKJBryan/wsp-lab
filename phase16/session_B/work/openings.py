"""Opening lines and outcome per pass: does a run's closing rate depend on a run-wide factor? CPU.
usage: python openings.py <pass> [<pass> ...]
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402


def main():
    for k in [int(x) for x in sys.argv[1:]]:
        rows = R.gen_rows(k)
        first = collections.Counter()
        by = collections.defaultdict(lambda: [0, 0])
        lens = []
        for pid, r in rows.items():
            t = r["text"]
            head = t[:60].strip().split("\n")[0][:48]
            first[head] += 1
            by[head][0] += 1
            by[head][1] += int(r["closed"])
            lens.append(r["n_tokens"])
        print(f"pass {k}: closed {sum(1 for r in rows.values() if r['closed'])}, correct {sum(1 for r in rows.values() if r['ok_boxed'])}")
        for head, n in first.most_common(8):
            print(f"   {n:3d} traces start with {head!r:52s}: closed {by[head][1]}")


if __name__ == "__main__":
    main()
