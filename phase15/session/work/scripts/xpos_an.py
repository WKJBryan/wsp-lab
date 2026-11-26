"""Detector analysis in x-space (x = post_attention_layernorm output = what an MLP neuron sees).

Uses out/xpos_1.pt (all traces, all positions, layers 13 and 14).
  detector : mean difference (LOOP line starts - FRESH line starts), trained on odd-id traces,
             evaluated on even-id traces over ALL positions by region (early / pre / loop)
  constant : a direction w with x.w as constant as possible over fresh positions, to realise the
             threshold (no bias term in the MLP): (a) the mean xbar, (b) shrinkage direction
             (Sigma + lambda I)^-1 xbar  [closed form], (c) the best single coordinate.
Usage: python xpos_an.py
"""
import sys, torch
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
layers, data = X['layers'], X['data']
keys = sorted(data, key=lambda s: int(s[1:]))
train = [k for i, k in enumerate(keys) if i % 2 == 1]      # odd ids
test = [k for i, k in enumerate(keys) if i % 2 == 0]       # even ids
NORM_EPS = 1e-6


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


def q(v, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    return ' '.join(f'{float(torch.quantile(v, p)):+.3f}' for p in ps)


for li, L in enumerate(layers):
    print(f'\n================ layer {L} (x = MLP input) ================')
    # ---- constant direction (from TRAIN, early + closed positions)
    xs_tr = []
    for k in train:
        D = data[k]
        x = D['x'][li].float(); T = x.shape[0]
        reg = regions_of(D, T)
        e = (reg == 0)
        xs_tr.append(x[e][::3])
    xs_tr = torch.cat(xs_tr)
    xbar = xs_tr.mean(0)
    mu, sd = xs_tr.mean(0), xs_tr.std(0)
    best_coord = int((sd / mu.abs()).argmin())
    sub = xs_tr[torch.randperm(xs_tr.shape[0])[:120000]]
    C = torch.cov(sub.t())
    w_sh = torch.linalg.solve(C + 0.05 * torch.trace(C) / 1024 * torch.eye(1024), mu)
    print(f'  |x| mean {float(xs_tr.norm(dim=1).mean()):.1f}; |xbar| {float(xbar.norm()):.2f} (|xbar|/|x| = {float(xbar.norm() / xs_tr.norm(dim=1).mean()):.2f}); best constant coordinate: {best_coord} (mean {float(mu[best_coord]):.3f}, rel-std {float(sd[best_coord] / mu[best_coord]):.3f})')
    cands = {'xbar': xbar / xbar.dot(xbar), 'shrink': w_sh / w_sh.dot(mu)}
    cands[f'coord{best_coord}'] = None
    for name, w in cands.items():
        line = f'  constant via {name:12s}: projection (train normalised to 1) | by region [p1 p5 p25 p50 p75 p95 p99]:'
        for reg_name, reg_id in [('early', 0), ('pre', 1), ('loop', 2)]:
            vals = []
            for k in test:
                D = data[k]
                x = D['x'][li].float(); T = x.shape[0]
                reg = regions_of(D, T)
                sel = reg == reg_id
                if not sel.any():
                    continue
                p = x[sel][:, best_coord] / mu[best_coord] if w is None else x[sel] @ w
                vals.append(p)
            line += f'\n        {reg_name:5s}: {q(torch.cat(vals))}'
        print(line)
    # ---- detector in x-space (train on odd ids)
    pos, neg = [], []
    for k in train:
        D = data[k]
        x = D['x'][li].float(); T = x.shape[0]
        reg = regions_of(D, T)
        lsm = D["ls"][:T]
        if (reg == 2).any():
            pos.append(x[(reg == 2) & lsm])
        neg.append(x[(reg == 0) & lsm][::2])
    d = torch.cat(pos).mean(0) - torch.cat(neg).mean(0)
    s_tr = torch.cat(pos) @ d
    thr = [float(torch.quantile(s_tr, qq)) for qq in (0.25, 0.50, 0.75, 0.90)]
    print(f'  detector |d| = {float(d.norm()):.3f}; TRAIN loop line-start scores q25/q50/q75/q90 = ' + ' '.join(f'{t:+.3f}' for t in thr))
    tot = {r: [0] * 6 for r in range(3)}
    per = {r: [] for r in range(3)}
    for k in test:
        D = data[k]
        x = D['x'][li].float(); T = x.shape[0]
        s = x @ d
        reg = regions_of(D, T)
        lsm = D["ls"][:T]
        for r in range(3):
            sel = reg == r
            if not sel.any():
                continue
            tot[r][0] += int(sel.sum()); tot[r][1] += int(lsm[sel].sum())
            for j, t in enumerate(thr):
                tot[r][2 + j] += int((s[sel] > t).sum())
            per[r].append((float(s[sel].max()), k, int(((s[sel] > thr[1]) & lsm[sel]).sum())))
    for r, name in enumerate(['early', 'pre', 'loop']):
        n, nls, a25, a50, a75, a90 = tot[r]
        print(f'     {name:5s}: positions {n:7d} (line starts {nls})  above q25 {a25:6d} ({100*a25/max(n,1):.2f}%)  q50 {a50:5d} ({100*a50/max(n,1):.3f}%)  q75 {a75:5d} ({100*a75/max(n,1):.3f}%)  q90 {a90:4d} ({100*a90/max(n,1):.3f}%)')
    for r, name in enumerate(['early', 'pre', 'loop']):
        top = sorted(per[r], reverse=True)[:12]
        print(f'     highest {name} maxima (trace:score:line-starts-above-q50): ' + ' '.join(f'{k}:{v:+.2f}:{c}' for v, k, c in top))
