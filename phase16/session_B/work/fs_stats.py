"""Conclusion-statement line ends ("FS rows") across the pass: how many, where, and how far </think> was.
usage: python fs_stats.py <mlp prefix> <full prefix> <pass> [--show]
"""
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
    mpre, fpre, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    show = "--show" in sys.argv
    rows = R.gen_rows(k)
    meta = {p["id"]: p for p in R.K.PRACTICE}
    n_with = 0
    allgap = []
    firsts = []
    for pid in rows:
        m = np.load(f"{mpre}.{pid}.npz")
        f = np.load(f"{fpre}.{pid}.npz")
        mm = m["meta"]
        fs = mm[:, 3] == 1
        g = f["gids"]
        text = "".join(STRS[i] for i in g)
        off = np.cumsum([0] + [len(STRS[i]) for i in g])
        if fs.any():
            n_with += 1
            firsts.append(mm[fs][0, 0])
        for r in mm[fs]:
            t = int(r[0])
            before = text[:off[t - 1]]
            line = before.split("\n")[-2] if len(before.split("\n")) >= 2 else ""
            think = f["prof_think"][t]
            top = f["prof_topl"][t, 0]
            topid = f["prof_top"][t, 0]
            gap = top - 1.5 - think          # after the presence penalty (the top token has been seen)
            allgap.append(gap)
            if show:
                print(f"{pid} L{meta[pid]['level'][-1]} t={t:5d} W={int(r[1]):4d} </think>={think:5.2f} top={STRS[topid]!r}:{top:5.2f} gap={gap:5.2f} "
                      f"next={STRS[int(r[2])]!r} | {line.strip()[-100:]!r}")
    allgap = np.array(allgap)
    print(f"\nsequences with >=1 conclusion line in the first 8000 tokens: {n_with}/{len(rows)}")
    print("first conclusion row per such sequence (tokens):", sorted(int(x) for x in firsts))
    print(f"gaps (top-1 after penalty minus </think>): n={len(allgap)} min {allgap.min():.2f} 10% {np.percentile(allgap, 10):.2f} "
          f"median {np.median(allgap):.2f} 90% {np.percentile(allgap, 90):.2f} max {allgap.max():.2f}")


if __name__ == "__main__":
    main()
