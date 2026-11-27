"""Closure and accuracy by batch position (first/second half), with level composition. CPU.
usage: python halves.py <pass> [<pass> ...]
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


for k in sys.argv[1:]:
    rows = R.gen_rows(k)
    ids = list(rows)
    half = len(ids) // 2
    for name, sel in (("first half", ids[:half]), ("second half", ids[half:])):
        lv = Counter(meta[i]["level"] for i in sel)
        ty = Counter(meta[i]["type"] for i in sel)
        cl = sum(rows[i]["closed"] for i in sel)
        ok = sum(rows[i]["ok_boxed"] for i in sel)
        print(f"pass {k} {name}: closed {cl}/{len(sel)} correct {ok}/{len(sel)} | levels {dict(sorted(lv.items()))} | "
              f"types {dict(ty.most_common(4))}")
    # closure by level in each half
    for lv in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
        s = []
        for name, sel in (("first", ids[:half]), ("second", ids[half:])):
            sub = [i for i in sel if meta[i]["level"] == lv]
            s.append(f"{name}: {sum(rows[i]['closed'] for i in sub)}/{len(sub)} closed, {sum(rows[i]['ok_boxed'] for i in sub)} ok")
        print(f"     {lv}: " + " | ".join(s))
    a = sum(rows[i]["closed"] for i in ids[:half]); c = sum(rows[i]["closed"] for i in ids[half:])
    print(f"     closed first {a}/{half} vs second {c}/{len(ids) - half}: Fisher p = {fisher(a, half - a, c, len(ids) - half - c):.4f}")
