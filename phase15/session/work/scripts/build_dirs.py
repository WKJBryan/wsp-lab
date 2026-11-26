"""Build the final ingredients of the graft from ALL traces and tabulate candidate thresholds.

Direction d: mean(x at LOOP line starts) - mean(x at FRESH line starts), layer L, all traces.
Constant direction w: (Sigma + 0.05 tr(Sigma)/1024 I)^-1 xbar over fresh positions, normalised so
that mean(x.w) = 1 on those positions.
Note: with d trained on all traces the table below is in-sample; generalisation was checked in
xpos_an.py (train odd ids / test even ids).

Usage: python build_dirs.py --layer 14   -> saves out/graft_dirs_L14.pt
"""
import sys, torch, argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--fresh_every', type=int, default=3)
a = ap.parse_args()
L = a.layer
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
layers, data = X['layers'], X['data']
li = layers.index(L)
keys = sorted(data, key=lambda s: int(s[1:]))
looping = {k: bool(data[k]['loop_start']) for k in keys}


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


xs, lss, regs = {}, {}, {}
for k in keys:
    D = data[k]
    x = D['x'][li].float(); T = x.shape[0]
    xs[k], lss[k], regs[k] = x, D['ls'][:T], regions_of(D, T)
# ---- detector
pos = torch.cat([xs[k][(regs[k] == 2) & lss[k]] for k in keys if looping[k]])
neg = torch.cat([xs[k][(regs[k] == 0) & lss[k]] for k in keys])
d = pos.mean(0) - neg.mean(0)
# ---- constant direction
fr = torch.cat([xs[k][regs[k] == 0][::a.fresh_every] for k in keys])
mu = fr.mean(0)
sub = fr[torch.randperm(fr.shape[0])[:120000]]
C = torch.cov(sub.t())
w = torch.linalg.solve(C + 0.05 * torch.trace(C) / 1024 * torch.eye(1024), mu)
w = w / w.dot(mu)                       # mean(x.w) = 1 on fresh positions
print(f'layer {L}: |d| = {float(d.norm()):.2f}; fresh |x| mean {float(fr.norm(dim=1).mean()):.1f}; '
      f'constant projection on fresh: p1 {float(torch.quantile(fr @ w, 0.01)):.3f} p99 {float(torch.quantile(fr @ w, 0.99)):.3f}')
S = {k: xs[k] @ d for k in keys}
P = {k: xs[k] @ w for k in keys}
s_loop = torch.cat([S[k][(regs[k] == 2) & lss[k]] for k in keys if looping[k]])
qs = {q: float(torch.quantile(s_loop, q)) for q in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}
print('loop line-start score quantiles: ' + ' '.join(f'q{int(q*100)}={v:.0f}' for q, v in qs.items()))
print('theta (quantile) | early positions above (traces with any) | pre-loop above | loop above (positions, traces with any of N looping)')
for q, thr in qs.items():
    e = [(k, int(((regs[k] == 0) & (S[k] > thr)).sum())) for k in keys]
    e = [(k, c) for k, c in e if c]
    pre = sum(int(((regs[k] == 1) & (S[k] > thr)).sum()) for k in keys)
    lp = sum(int(((regs[k] == 2) & (S[k] > thr)).sum()) for k in keys if looping[k])
    lt = sum(1 for k in keys if looping[k] and ((regs[k] == 2) & (S[k] > thr)).any())
    nlp = sum(int((regs[k] == 2).sum()) for k in keys if looping[k])
    print(f'  {thr:6.0f} (q{int(q*100):2d}) | {sum(c for _, c in e):6d} in {len(e):2d} traces: {" ".join(f"{k}:{c}" for k, c in sorted(e, key=lambda t: -t[1])[:8])} | {pre:5d} | {lp:6d}/{nlp} ({100*lp/nlp:.1f}%), {lt}/{sum(looping.values())} traces')
# constant-projection stability by region (all traces)
for name, rid in [('early', 0), ('pre', 1), ('loop', 2)]:
    v = torch.cat([P[k][regs[k] == rid] for k in keys if (regs[k] == rid).any()])
    print(f'  constant projection {name:5s}: p1 {float(torch.quantile(v, 0.01)):.3f} p50 {float(v.median()):.3f} p99 {float(torch.quantile(v, 0.99)):.3f} min {float(v.min()):.3f} max {float(v.max()):.3f}')
# exceedance distribution in loops at q75: how far above the threshold do loop positions go?
thr = qs[0.75]
exc = torch.cat([S[k][(regs[k] == 2) & (S[k] > thr)] - thr for k in keys if looping[k]])
print(f'  exceedance over q75 threshold in loops: n={exc.numel()} q10 {float(torch.quantile(exc, 0.1)):.0f} q50 {float(exc.median()):.0f} q90 {float(torch.quantile(exc, 0.9)):.0f} max {float(exc.max()):.0f}')
# per-trace loop maxima (first 800 loop tokens), lowest first
lm = sorted((float(S[k][regs[k] == 2].max()), k) for k in keys if looping[k])
print('  lowest in-loop maxima (trace:score): ' + ' '.join(f'{k}:{v:.0f}' for v, k in lm[:16]))
torch.save(dict(layer=L, d=d, w=w, thr_loop_ls=qs), f'/home/user/wsp14/work/out/graft_dirs_L{L}.pt')
print(f'saved out/graft_dirs_L{L}.pt')
