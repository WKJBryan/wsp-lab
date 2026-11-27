"""Search for a direction whose projection on the last-MLP input is (nearly) constant across positions.
usage: python probe3.py
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
    Xs, regs = [], []
    for pid in pids[::2]:
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        Xs.append(x[::3]); regs.append(region[::3])
    X = np.concatenate(Xs).astype(np.float32); Rg = np.concatenate(regs)
    mu = X.mean(0); sd = X.std(0)
    cv = sd / (np.abs(mu) + 1e-6)
    order = np.argsort(cv)[:15]
    print("dimensions with the most stable value (lowest std/|mean|):")
    for k in order:
        print(f"   dim {k:4d}: mean {mu[k]:8.3f} std {sd[k]:.4f} cv {cv[k]:.4f}")
    # PCA top components: how stable are their projections?
    Xc = X - mu
    cov = (Xc.T @ Xc) / len(Xc)
    w, V = np.linalg.eigh(cov)
    print("\ntop principal components: share of variance, relative std of the projection, mean projection")
    for i in range(1, 8):
        v = V[:, -i]
        p = X @ v
        print(f"   PC{i}: var share {w[-i] / w.sum():.3f}; mean proj {p.mean():8.3f}; std {p.std():.3f}; cv {p.std() / abs(p.mean()):.3f}")
    # combination: regress the constant 1 on x (least squares through the origin) -> best constant direction
    # closed-form ridge: o = (X^T X + lam I)^-1 X^T 1
    lam = 1e-2 * np.trace(cov)
    o = np.linalg.solve(cov + lam * np.eye(1024), mu)
    p = X @ o
    print(f"\nleast-squares constant direction: projection 1/10/50/90/99% {np.percentile(p, [1, 10, 50, 90, 99]).round(3)}")
    for rg, nm in ((0, "prompt"), (1, "thinking"), (2, "answer")):
        m = Rg == rg
        if m.sum():
            print(f"   {nm}: median {np.median(p[m]):.3f} 1% {np.percentile(p[m], 1):.3f} 99% {np.percentile(p[m], 99):.3f}")
    np.save(WORK / "r1/const_dir.npy", o)


if __name__ == "__main__":
    main()
