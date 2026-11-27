"""Look at the negatives with the highest probe scores at line ends in the thinking region.
usage: python probe_neg.py [N] [lo_score] [--all]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else -1e9
    show_all = "--all" in sys.argv
    pids = list(R.gen_rows(1))
    pr = np.load(WORK / "r1/probe.npz")
    c = pr["c"] / np.linalg.norm(pr["c"])
    rows = []
    for pi, pid in enumerate(pids):
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        s = x @ c
        nl = np.array(["\n" in PR.STRS[i] for i in ids])
        g = ids[plen:]
        W = np.cumsum(np.array([PR.STRS[i] for i in g]).__array__() != "")  # placeholder
        isw = np.array([pr_ for pr_ in PR.STRS])  # unused
        wait = np.array([pr_s.lstrip("* ").startswith("Wait") for pr_s in PR.STRS])
        wc = np.cumsum(wait[g]) - wait[g]
        for p in np.where(nl & (region == 1) & (cls == 0))[0]:
            if s[p] >= lo:
                t = p - plen + 1
                lines = "".join(PR.STRS[i] for i in ids[max(0, p - 120):p + 1]).split("\n")
                line = [l for l in lines[:-1] if l.strip()][-1].strip() if len(lines) > 1 else ""
                rows.append((s[p], pid, int(meta[3]), t, wc[min(t, len(wc) - 1)], line[-110:]))
    rows.sort(key=lambda r: -r[0])
    print(f"{len(rows)} line ends (not labelled commit) in thinking with score >= {lo}")
    for r in rows[:N] if not show_all else rows:
        print(f"{r[0]:5.2f} {r[1]:4s} L{r[2]} t={r[3]:5d} W={r[4]:4d} | {r[5]!r}")


if __name__ == "__main__":
    main()
