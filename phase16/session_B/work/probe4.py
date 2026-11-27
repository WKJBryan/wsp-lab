"""Line-end-only design of the conclusion readout and the constant reference (CPU).
Fits, in closed form (ridge least squares, no iteration):
  o  : direction whose projection is as constant as possible over line ends in the thinking,
  c  : readout separating commit line ends from all other line ends (diff of means, or ridge for
       several strengths, judged on the other half of the sequences).
Reports trigger rates of  z = c.x - theta * o.x  for several theta.
usage: python probe4.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def gather(pids):
    X, C, Rg, NL, PID = [], [], [], [], []
    for pi, pid in enumerate(pids):
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        X.append(x); C.append(cls); Rg.append(region)
        NL.append(np.array(["\n" in PR.STRS[i] for i in ids])); PID.append(np.full(len(ids), pi))
    return np.concatenate(X), np.concatenate(C), np.concatenate(Rg), np.concatenate(NL), np.concatenate(PID)


def ridge_const(X, lam_rel):
    cov = X.T @ X / len(X)
    lam = lam_rel * np.trace(cov)
    return np.linalg.solve(cov + lam * np.eye(X.shape[1]), X.sum(0) / len(X))


def ridge(X, y, lam_rel):
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / len(Xc)
    lam = lam_rel * np.trace(cov)
    return np.linalg.solve(cov + lam * np.eye(X.shape[1]), Xc.T @ (y - y.mean()) / len(Xc))


def main():
    pids = list(R.gen_rows(1))
    XA, CA, RA, NA, PA = gather(pids[::2])
    XB, CB, RB, NB, PB = gather(pids[1::2])
    nA, nB = NA.astype(bool), NB.astype(bool)
    L = XA[nA & (RA == 1)][::2]
    o = ridge_const(L, 1e-3)
    o /= float(np.median(L @ o))
    for X, nl, Rg, nm in ((XA, nA, RA, "fit half"), (XB, nB, RB, "other half")):
        for rg, rname in ((1, "thinking line ends"), (0, "prompt line ends"), (2, "answer-phase line ends")):
            m = nl & (Rg == rg)
            if m.sum():
                p = X[m] @ o
                print(f"constant reference on {nm}, {rname}: n={m.sum()} 1/10/50/90/99% {np.percentile(p, [1, 10, 50, 90, 99]).round(3)}")
    posA = nA & (RA == 1) & (CA == 1); negA = nA & ~posA
    posB = nB & (RB == 1) & (CB == 1); negB = nB & ~posB
    cands = {"diff-of-means (all positions)": None}
    for lam in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        cands[f"ridge line ends lam {lam}"] = lam
    pr = np.load(WORK / "r1/probe_all.npz")
    results = {}
    for name, lam in cands.items():
        if lam is None:
            c = pr["c"].copy()
        else:
            c = ridge(np.concatenate([XA[posA], XA[negA]]), np.concatenate([np.ones(posA.sum()), np.zeros(negA.sum())]), lam)
            c /= np.linalg.norm(c)
        cA = c * (25.0 / float((XA[posA] @ c).mean()))
        cB = c * (25.0 / float((XB[posB] @ c).mean()))       # same normalisation rule applied with each half's own statistics (diagnostic only)
        sA, sB = XA @ cA, XB @ cB
        print(f"\n== {name}: positives (fit half scaled to mean 25) 10/50/90% {np.percentile(sA[posA], [10, 50, 90]).round(2)}; "
              f"other half scaled with its own positives: {np.percentile(sB[posB], [10, 50, 90]).round(2)}")
        # honest: apply the fit-half scaling to the other half
        cB2 = c * (25.0 / float((XA[posA] @ c).mean()))
        sB2 = XB @ cB2
        print(f"   other half with the fit-half scaling: positives 10/50/90% {np.percentile(sB2[posB], [10, 50, 90]).round(2)}")
        for b in (22, 24, 25, 26, 27, 28, 30):
            print(f"   score >= {b:2d} (other half): commit {np.mean(sB2[posB] >= b):.3f} | other thinking line ends "
                  f"{np.mean(sB2[nB & (RB == 1) & ~posB] >= b):.4f} | non-line-end thinking {np.mean(sB2[(~nB) & (RB == 1)] >= b):.5f} | "
                  f"answer phase {np.mean(sB2[RB == 2] >= b):.4f} | prompt {np.mean(sB2[RB == 0] >= b):.4f}")
        results[name] = (cA, cB2)
    # effective trigger with the constant reference for the diff-of-means probe
    cA, cB2 = results["diff-of-means (all positions)"]
    zA = XA @ cA - XA @ o; zB = XB @ cB2 - XB @ o
    print("\nthreshold through the constant reference, diff-of-means probe: trigger when c.x - theta*rho >= k")
    for theta in (22, 23, 24, 25, 26):
        row = []
        for k in (1, 2, 3):
            mA = zA >= (theta + k - 1); mB = zB >= (theta + k - 1)
            row.append(f"k={k}: commit {np.mean(mB[posB]):.3f} other {np.mean(mB[nB & (RB == 1) & ~posB]):.4f} "
                       f"nonLE {np.mean(mB[(~nB) & (RB == 1)]):.5f} ans {np.mean(mB[RB == 2]):.4f}")
        print(f"  theta {theta}: " + " | ".join(row))
    np.savez(WORK / "r1/probe_le.npz", c=pr["c"], o=o)


if __name__ == "__main__":
    main()
