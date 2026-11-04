"""Phase-2 editor analysis: identify each net's task from weights alone.

Allowed (protocol.md): any offline computation on the tensors, synthetic-input
probes, comparison to a fresh reference init. NO env, NO training data, NO sealed/.

Usage: uv run python phase2/editor/analyze.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "phase1"))
from policy import make_policy  # noqa: E402

NETS = HERE.parent / "nets"
torch.manual_seed(999)
REF = make_policy()  # fresh init reference


def load(name):
    p = make_policy()
    p.load_state_dict(torch.load(NETS / name, map_location="cpu", weights_only=True))
    p.eval()
    return p


def diff(p, g):
    with torch.no_grad():
        z = torch.relu(p[0](g))
        z = torch.relu(p[2](z))
        lg = p[4](z)
    return lg[:, 1] - lg[:, 0]


def analyze(name):
    p = load(name)
    T = p.state_dict()
    out = {"name": name}

    # norms vs fresh init
    out["norm0"] = float(T["0.weight"].norm())
    out["norm2"] = float(T["2.weight"].norm())
    out["norm4"] = float(T["4.weight"].norm())
    out["ref_norm0"] = float(REF.state_dict()["0.weight"].norm())

    # feature usage: layer-0 column norms (normalized to mean 1)
    col = T["0.weight"].norm(dim=0)
    out["colnorms"] = [round(float(c / col.mean()), 2) for c in col]

    # sensitivity: variance of diff when each feature is jittered
    g = torch.Generator().manual_seed(7)
    S = torch.randn(2048, 4, generator=g)
    d0 = diff(p, S)
    sens = []
    for i in range(4):
        S2 = S.clone()
        S2[:, i] += 0.5
        sens.append(float((diff(p, S2) - d0).pow(2).mean().sqrt()))
    m = max(sens) + 1e-9
    out["sens"] = [round(s / m, 2) for s in sens]

    # linear additivity fit on N(0,1)^4: closed-form lstsq of diff on inputs
    X = torch.cat([S, torch.ones(S.shape[0], 1)], dim=1)
    sol = torch.linalg.lstsq(X, d0.unsqueeze(1)).solution.squeeze(1)
    pred = X @ sol
    ss_res = float((pred - d0).pow(2).sum())
    ss_tot = float((d0 - d0.mean()).pow(2).sum())
    out["linfit_r2"] = round(1 - ss_res / ss_tot, 3)

    # same fit on WIDE-range inputs (extrapolation stability)
    S_w = torch.randn(2048, 4, generator=g) * 8
    d_w = diff(p, S_w)
    X_w = torch.cat([S_w, torch.ones(S_w.shape[0], 1)], dim=1)
    solw = torch.linalg.lstsq(X_w, d_w.unsqueeze(1)).solution.squeeze(1)
    r2w = 1 - float((X_w @ solw - d_w).pow(2).sum()) / float((d_w - d_w.mean()).pow(2).sum())
    out["linfit_wide_r2"] = round(r2w, 3)

    # roughness: sign flips of decision along random unit-metric lines
    flips, dist = 0, 0.0
    for k in range(64):
        a = torch.randn(4, generator=g)
        b = torch.randn(4, generator=g)
        ts = torch.linspace(0, 1, 33)
        pts = a + ts.unsqueeze(1) * (b - a)
        s = torch.sign(diff(p, pts))
        flips += int((s[1:] != s[:-1]).sum())
        dist += float((b - a).norm())
    out["flips_per_unit"] = round(flips / dist, 2)

    # dead units on standard-normal grid
    with torch.no_grad():
        z = torch.relu(p[0](S))
        z = torch.relu(p[2](z))
    out["dead2"] = int(((z == 0).float().mean(0) > 0.99).sum())

    return out


hdr = f"{'net':4s} {'|W0|':>6s} {'|W2|':>7s} {'|W4|':>6s} {'cols(4)':>18s} {'sens(4)':>16s} {'r2':>6s} {'r2wide':>7s} {'flips/u':>8s} {'dead2':>5s}"
print(hdr)
print(f"{'init':4s} {float(REF.state_dict()['0.weight'].norm()):6.2f}   (fresh reference: |W0| at initialization)")
for n in "ABCDE":
    o = analyze(f"{n}.pt")
    print(
        f"{o['name'][:4]:4s} {o['norm0']:6.2f} {o['norm2']:7.2f} {o['norm4']:6.2f} "
        f"{str(o['colnorms']):>18s} {str(o['sens']):>16s} {o['linfit_r2']:6.3f} "
        f"{o['linfit_wide_r2']:7.3f} {o['flips_per_unit']:8.2f} {o['dead2']:5d}"
    )
