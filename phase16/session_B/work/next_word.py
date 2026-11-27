"""CPU: the conclusion score at the position where the next line's first word is decided, and what word
the model itself writes after a line that states the answer.
usage: python next_word.py
"""
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"
WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def main():
    pids = list(R.gen_rows(1))
    pr = np.load(WORK / "cands/E2.npz")
    c, o = pr["c"], pr["o"]
    S, RHO, kind, words, tokids = [], [], [], [], []
    for pid in pids:
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        s = x @ c; rho = x @ o
        for q in np.where((cls == 1) & (region == 1))[0]:
            # first word token of the next line
            w = q + 1
            while w < len(ids) and ids[w] < len(PR.STRS) and not PR.STRS[ids[w]].strip():
                w += 1
            if w >= len(ids) or w - q > 4:
                continue
            st = PR.STRS[ids[w]].strip()
            if not WORD.match(st) and st not in ("*", "$", "-"):
                continue
            S.append(s[w - 1]); RHO.append(rho[w - 1]); kind.append("commit"); words.append(st); tokids.append(int(ids[w]))
        # all other line-start word decisions in the thinking (sampled)
        th = np.where(region == 1)[0]
        for q in th[::7]:
            if q + 1 >= len(ids):
                continue
            st = PR.STRS[ids[q]].strip()
            if not st and q >= 1:
                w = q + 1
                while w < len(ids) and not PR.STRS[ids[w]].strip():
                    w += 1
                if w < len(ids) and WORD.match(PR.STRS[ids[w]].strip()) and cls[q - 1] != 1:
                    S.append(s[q]); RHO.append(rho[q]); kind.append("other"); words.append(PR.STRS[ids[w]].strip()); tokids.append(int(ids[w]))
    S = np.array(S); RHO = np.array(RHO); kind = np.array(kind)
    z = S - 25.0 * RHO
    cm = kind == "commit"
    print(f"word decisions after a commit line: n={cm.sum()}; score c.x quantiles 10/50/90: {np.percentile(S[cm], [10, 50, 90]).round(2)}; "
          f"z>=1: {np.mean(z[cm] >= 1):.3f}, z>=2: {np.mean(z[cm] >= 2):.3f}")
    print(f"other line-start word decisions (sample): n={(~cm).sum()}; score quantiles 50/99/99.9/max: "
          f"{np.percentile(S[~cm], [50, 99, 99.9, 100]).round(2)}; z>=1: {np.mean(z[~cm] >= 1):.4f}, z>=2: {np.mean(z[~cm] >= 2):.4f}")
    print("first word the model itself writes on the line after a commit line:")
    cnt = collections.Counter(np.array(words)[cm])
    for w_, n_ in cnt.most_common(25):
        print(f"   {n_:3d} {w_!r}")
    print("token ids of 'Final' variants:", collections.Counter(int(t) for t, k in zip(tokids, kind) if k == "commit" and PR.STRS[t].strip() == "Final"))


if __name__ == "__main__":
    main()
