"""Phase 1 final measurements (harness-side, outside the editor's 512-episode budget).

P   primary endpoint: chosen candidate over 256 episodes, seeds 900000-900255
C1  untouched target, same protocol
C2  random-edit control: 20 trials, Gaussian noise with Frobenius norm matched
    to the editor's final edit delta; 64 episodes each (seeds 910000+); control
    value = BEST of 20 (hardest version, per PROTOCOL.md)
C3  retrain upper bound: fresh seeds 100-102, 1500 episodes each, primary protocol

Usage: uv run python phase1/controls.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate import evaluate
from train import train_one

WINNER = HERE / "editor" / "cand_v5.pt"


def arr_str(a):
    return f"mean={a.mean():.1f} std={a.std():.1f} min={a.min():.0f} max={a.max():.0f}"


print("== PRIMARY ENDPOINT: cand_v5, 256 eps, seeds 900000+ ==")
p = evaluate(str(WINNER), 256, 900000)
print("cand_v5:", arr_str(p))

print("\n== C1: untouched target, same protocol ==")
c1 = evaluate(str(HERE / "target.pt"), 256, 900000)
print("target :", arr_str(c1))

print("\n== C2: random-edit control (20 trials, matched norm, best-of-20) ==")
T = torch.load(HERE / "target.pt", map_location="cpu", weights_only=True)
E = torch.load(WINNER, map_location="cpu", weights_only=True)
delta_sq = sum(((E[k].float() - T[k].float())).norm() ** 2 for k in T)
D = delta_sq.sqrt().item()
print(f"editor edit Frobenius norm: {D:.4f}")
means = []
tmp = HERE / "editor" / "c2_trial.pt"
for t in range(20):
    g = torch.Generator().manual_seed(7000 + t)
    noise = {k: torch.randn(v.shape, generator=g) for k, v in T.items()}
    nrm = sum(n.norm().item() ** 2 for n in noise.values()) ** 0.5
    cand = {k: (v.float() + noise[k] * (D / nrm)).contiguous() for k, v in T.items()}
    torch.save(cand, tmp)
    a = evaluate(str(tmp), 64, 910000 + 64 * t)
    means.append(float(a.mean()))
tmp.unlink(missing_ok=True)
print("trial means:", [round(m, 1) for m in means])
print(f"C2 control value (best of 20): {max(means):.1f}")

print("\n== C3: retrain upper bound (fresh seeds 100-102, primary protocol) ==")
for s in [100, 101, 102]:
    pol, _ = train_one(s)
    ck = HERE / "ckpt" / f"seed{s}.pt"
    torch.save(pol.state_dict(), ck)
    a = evaluate(str(ck), 256, 900000)
    print(f"retrain seed {s}: {arr_str(a)}")
