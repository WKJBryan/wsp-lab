"""Score distribution of the conclusion probe at line ends and elsewhere (CPU).
usage: python probe_dist.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    pids = list(R.gen_rows(1))
    pr = np.load(WORK / "r1/probe.npz")
    c = pr["c"] / np.linalg.norm(pr["c"])
    bands = [18, 20, 21, 22, 23, 24, 25, 26, 28, 30]
    nl_commit = np.zeros(len(bands)); nl_other = np.zeros(len(bands)); nonl = np.zeros(len(bands))
    tot = dict(nl_commit=0, nl_other=0, nonl=0, first_above={}, first_commit={})
    samples = {b: [] for b in (22, 23, 24, 25)}
    nonl_samples = []
    for pi, pid in enumerate(pids):
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        s = x @ c
        nl = np.array(["\n" in PR.STRS[i] for i in ids])
        th = region == 1
        commit = nl & th & (cls == 1)
        other = nl & th & (cls == 0)
        isnl = (~nl) & th
        tot["nl_commit"] += commit.sum(); tot["nl_other"] += other.sum(); tot["nonl"] += isnl.sum()
        for b in bands:
            nl_commit[bands.index(b)] += (commit & (s >= b)).sum()
            nl_other[bands.index(b)] += (other & (s >= b)).sum()
            nonl[bands.index(b)] += (isnl & (s >= b)).sum()
        fa = np.where(th & (s >= 24))[0]
        tot["first_above"][pid] = int(fa[0] - plen) if len(fa) else None
        fc = np.where(commit)[0]
        tot["first_commit"][pid] = int(fc[0] - plen) if len(fc) else None
        for b in samples:
            for p in np.where(other & (s >= b) & (s < b + 1))[0][:1]:
                lines = "".join(PR.STRS[i] for i in ids[max(0, p - 100):p + 1]).split("\n")
                line = [l for l in lines[:-1] if l.strip()][-1].strip() if len(lines) > 1 else ""
                samples[b].append((pid, float(s[p]), line[-100:]))
        for p in np.where(isnl & (s >= 24))[0][:3]:
            nonl_samples.append((pid, float(s[p]), "".join(PR.STRS[i] for i in ids[max(0, p - 40):p + 1])[-90:].replace("\n", "⏎")))
    print("threshold | commit line ends above | other line ends above | non-line-end thinking positions above")
    for i, b in enumerate(bands):
        print(f"   {b:3d}   | {int(nl_commit[i]):6d} / {tot['nl_commit']} | {int(nl_other[i]):6d} / {tot['nl_other']} "
              f"({100 * nl_other[i] / tot['nl_other']:.3f}%) | {int(nonl[i]):6d} / {tot['nonl']} ({100 * nonl[i] / tot['nonl']:.4f}%)")
    fa = [v for v in tot["first_above"].values() if v is not None]
    print(f"\nsequences with a thinking position scoring >= 24 within 8000 tokens: {len(fa)}/128; first such position (tokens):",
          sorted(fa))
    for b in samples:
        print(f"\nsample other line ends with score in [{b}, {b + 1}):")
        for pid, sc, line in samples[b][:25]:
            print(f"   {sc:5.2f} {pid}: {line!r}")
    print("\nsample non-line-end thinking positions with score >= 24:")
    for pid, sc, t in nonl_samples[:40]:
        print(f"   {sc:5.2f} {pid}: ...{t!r}")


if __name__ == "__main__":
    main()
