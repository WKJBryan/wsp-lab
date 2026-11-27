"""CPU design checks for the stop-pathway neuron.
1. is there a direction whose projection is (nearly) constant across positions? (to build a threshold)
2. which tokens' readout rows lie close to that of </think>? (collateral of writing along it)
3. refit the conclusion direction on all sequences.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"


def main():
    pids = list(R.gen_rows(1))
    X, C, Rg, P, off = [], [], [], [], [0]
    for pi, pid in enumerate(pids):
        x, ids, plen, meta = PR.load(pid)
        cls, region = PR.label_positions(ids, plen)
        X.append(x); C.append(cls); Rg.append(region); P.append(np.full(len(ids), pi)); off.append(off[-1] + len(ids))
    X = np.concatenate(X); C = np.concatenate(C); Rg = np.concatenate(Rg); P = np.concatenate(P)
    pos = (C == 1) & (Rg == 1)
    xbar = X.mean(0)
    proj = X @ xbar / float(xbar @ xbar)
    print(f"mean direction: |xbar| {np.linalg.norm(xbar):.2f}; projection xbar.x/|xbar|^2 over positions: "
          f"1/10/50/90/99% {np.percentile(proj, [1, 10, 50, 90, 99]).round(3)}; |x| median {np.median(np.linalg.norm(X, axis=1)):.2f}")
    # refit on all
    c = X[pos].mean(0) - X[~pos].mean(0)
    c /= np.linalg.norm(c)
    s = X @ c
    nl = np.array(["\n" in PR.STRS[i] for i in np.concatenate([np.load(WORK / f"r1/x.{pid}.npz")["ids"] for pid in pids])])
    th = Rg == 1
    for b in (22, 23, 24, 25, 26, 27, 28):
        print(f"  score >= {b}: commit line ends {np.mean(s[pos] >= b):.3f} | other line ends in thinking {np.mean(s[th & nl & ~pos] >= b):.4f} "
              f"| non-line-end thinking positions {np.mean(s[th & ~nl] >= b):.6f} | answer phase {np.mean(s[Rg == 2] >= b):.4f} | prompt {np.mean(s[Rg == 0] >= b):.5f}")
    np.savez(WORK / "r1/probe_all.npz", c=c, xbar=xbar)
    # 2. readout neighbours of </think>
    with safe_open(sorted((R.K.model_dir()).glob("*.safetensors"))[0], framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float()
        nw = f.get_tensor("model.language_model.norm.weight").float()
    U = E * (1.0 + nw)[None, :]
    u = U[R.THINK_END]
    cos = (U @ u) / (U.norm(dim=1, keepdim=True)[:, 0] * u.norm())
    top = torch.topk(cos, 25)
    print("\ntokens whose readout is closest (cosine) to </think>'s:")
    for v, i in zip(top.values, top.indices):
        print(f"   {PR.STRS[int(i)]!r}: cos {float(v):.3f}  |row| {float(U[int(i)].norm()):.2f}")
    print(f"|U[</think>]| {float(u.norm()):.2f}; median |U[t]| {float(U.norm(dim=1).median()):.2f}")


if __name__ == "__main__":
    main()
