"""Offline evaluation of the complete grafted-neuron response on all collected positions.

neuron:  a = silu((x.d - theta * x.w) / s) * (x.w)          (gate x constant factor)
write :  a * c * v  added to the residual (v = unit output direction of </think>)
A position 'flips' when c*a >= FLIP (write strong enough that </think> wins).

Usage: python graft_eval.py --theta 445 --s 20 --c 5.5 [--layer 14] [--flip 3.5]
"""
import sys, torch, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--theta', type=float, default=445)
ap.add_argument('--s', type=float, default=20)
ap.add_argument('--c', type=float, default=5.5)
ap.add_argument('--flip', type=float, default=3.5)
ap.add_argument('--wmode', default='shrunk', help="shrunk | full : covariance estimator for the constant direction")
a = ap.parse_args()
G = torch.load(f'/home/user/wsp14/work/out/graft_dirs_L{a.layer}.pt')
L, d, thr_ls = G['layer'], G['d'], G['thr_loop_ls']
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(L)
keys = sorted(X['data'], key=lambda s: int(s[1:]))


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


if a.wmode == 'shrunk':
    w = G['w']
else:
    fr = torch.cat([X['data'][k]['x'][li].float()[regions_of(X['data'][k], X['data'][k]['x'][li].shape[0]) == 0][::3] for k in keys])
    mu = fr.mean(0)
    C = torch.cov(fr[torch.randperm(fr.shape[0])[:120000]].t())
    w = torch.linalg.solve(C, mu); w = w / w.dot(mu)
    print('full-covariance constant direction: not recommended with 120k samples (near-singular); using anyway')
thr, s, c, flip = a.theta, a.s, a.c, a.flip
print(f'layer {L}: theta {thr:.0f} s {s} c {c} flip-level write {flip} (a >= {flip / c:.3f}); |d| {float(d.norm()):.2f}')
stats = {r: dict(n=0, flip=0, half=0, amax=0.0) for r in range(3)}
per_trace = {r: {} for r in range(3)}
first_hit = {}
for k in keys:
    D = X['data'][k]
    x = D['x'][li].float(); T = x.shape[0]
    reg = regions_of(D, T)
    pw = x @ w
    act = torch.nn.functional.silu((x @ d - thr * pw) / s) * pw
    for r in range(3):
        sel = reg == r
        if not sel.any():
            continue
        ar = act[sel]
        stats[r]['n'] += int(sel.sum())
        stats[r]['flip'] += int((c * ar >= flip).sum())
        stats[r]['half'] += int((c * ar >= flip / 2).sum())
        stats[r]['amax'] = max(stats[r]['amax'], float(ar.max()))
        if float(c * ar.max()) >= flip / 2:
            per_trace[r][k] = (int((c * ar >= flip).sum()), float(c * ar.max()))
    if D['loop_start'] and (reg == 2).any():
        al = act[reg == 2]
        hit = torch.nonzero(c * al >= flip).flatten()
        first_hit[k] = int(hit[0]) if hit.numel() else None
names = ['early', 'pre', 'loop']
for r in range(3):
    st = stats[r]
    print(f'  {names[r]:5s}: positions {st["n"]:7d}  half-strength-or-more {st["half"]:6d}  flip-level {st["flip"]:6d} ({100*st["flip"]/max(st["n"],1):.3f}%)  max write {st["amax"]*c:6.1f}')
for r in (0, 1):
    top = sorted(per_trace[r].items(), key=lambda kv: -kv[1][1])[:10]
    print(f'  {names[r]} traces with any half-strength write: ' + ' '.join(f'{k}:{n}@{mx:.1f}' for k, (n, mx) in top))
n_hit = sum(1 for v in first_hit.values() if v is not None)
print(f'  loops with a flip-level position in their first 800 tokens: {n_hit}/{len(first_hit)}')
nohit = [(k, X['data'][k]['loop_start']) for k, v in first_hit.items() if v is None]
print('  loops without (trace:loop_start): ' + ' '.join(f'{k}:{s}' for k, s in nohit))
