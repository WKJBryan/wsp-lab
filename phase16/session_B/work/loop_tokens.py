"""Which tokens do the remaining loops start with? (token-level periodic tails of one pass)
usage: python loop_tokens.py <pass> [--show]
"""
import collections
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]


def main():
    k = int(sys.argv[1])
    show = "--show" in sys.argv
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    first = collections.Counter()
    n_loop = 0
    for pid, r in rows.items():
        if r["closed"]:
            continue
        g = tok(r["text"], add_special_tokens=False).input_ids
        per = R.token_period(g, max_period=4000, min_reps=6)
        if per is None:
            first["<no exact token loop>"] += 1
            if show:
                print(f"{pid} L{meta[pid]['level'][-1]}: no exact loop; tail: {r['text'][-300:]!r}")
            continue
        n_loop += 1
        u = g[per["start"]:per["start"] + per["period"]]
        # first word token of the unit (skip leading whitespace-only tokens)
        words = [i for i in u if STRS[i].strip()]
        w = STRS[words[0]].strip() if words else ""
        first[w] += 1
        if show:
            print(f"{pid} L{meta[pid]['level'][-1]}: period {per['period']} tok x {per['reps']} | unit: {''.join(STRS[i] for i in u)[:150]!r}")
    print(f"\npass {k}: {n_loop} not-closed traces with an exact token loop")
    for w, c in first.most_common(40):
        print(f"   {c:3d}  {w!r}")


if __name__ == "__main__":
    main()
