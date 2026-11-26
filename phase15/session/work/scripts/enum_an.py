"""Leave-one-trace-out test of a runaway-enumeration detector direction (closed-form statistics).

For each held-out runaway trace: direction trained on the other 12 runaway traces + the fresh
samples of even-id traces; evaluated on (i) the held-out runaway trace, (ii) the fresh samples of
odd-id traces (never used in training). Two estimators: mean difference and ridge LDA.

Usage: python enum_an.py
"""
import sys, torch
torch.set_num_threads(8)
E = torch.load('/home/user/wsp14/work/out/enum_L14.pt', weights_only=False)
pos, neg = E['pos'], E['neg']
pkeys = sorted(pos)
nkeys = sorted(neg, key=lambda s: int(s[1:]))
negA = torch.cat([neg[k] for k in nkeys if int(k[1:]) % 2 == 0]).float()      # even ids: train side
negB = torch.cat([neg[k] for k in nkeys if int(k[1:]) % 2 == 1]).float()      # odd ids: test side
print(f'runaway traces {len(pkeys)}; fresh train {negA.shape[0]}, fresh test {negB.shape[0]}')


def direction(P, N, kind, lam_rel=0.01):
    mu_p, mu_n = P.mean(0), N.mean(0)
    d = mu_p - mu_n
    if kind == 'md':
        return d
    X = torch.cat([P, N]); Xc = X - X.mean(0)
    C = Xc.t() @ Xc / (X.shape[0] - 2)
    return torch.linalg.solve(C + lam_rel * float(torch.trace(C)) / 1024 * torch.eye(1024), d)


for kind, lam in [('md', 0), ('lda', 1e-3), ('lda', 1e-2), ('lda', 1e-1)]:
    res = []
    for hk in pkeys:
        P = torch.cat([pos[k]['x'] for k in pkeys if k != hk]).float()
        d = direction(P, negA, kind, lam)
        s_pos = pos[hk]['x'].float() @ d
        s_neg = negB @ d
        thr = float(s_neg.max())
        q99 = float(torch.quantile(s_neg, 0.99))
        frac = float((s_pos > thr).float().mean())
        # along the trace: samples are ordered in time; recall in the last 2000 tokens vs earlier
        n = s_pos.numel()
        res.append((hk, frac, float((s_pos[-500:] > thr).float().mean()), float(s_pos.median() - thr), float(s_pos.max() - thr), thr, q99))
    print(f'\n== {kind} lambda {lam:g}: held-out runaway trace | recall at zero fresh false alarm | recall in its last 2000 tokens | median margin over fresh max | max margin')
    for hk, frac, last, med, mx, thr, q99 in res:
        print(f'   {hk:9s} | {frac:.2f} | {last:.2f} | {med:+.1f} | {mx:+.1f}   (fresh max {thr:.1f}, fresh q99 {q99:.1f})')
    print(f'   mean recall at zero false alarm: {sum(r[1] for r in res)/len(res):.2f}; traces with recall >= 0.5: {sum(1 for r in res if r[1] >= 0.5)}/{len(res)}')
