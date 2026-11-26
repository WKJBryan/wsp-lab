"""Analysis of the line-start data (out/linestart_1.pt).

Q1: basic statistics by region (first token, top-1 probability, duplicate rate).
Q2: per layer, a mean-difference direction (train on half the traces) between LOOP line starts and
    FRESH ones (EARLY+LATE); on held-out traces: AUC, and the operating point where the threshold
    is the highest FRESH training score: recall on LOOP, false positives on EARLY and LATE.
Q3: the same restricted to lines whose first token is ' Wait' (content-matched).
Usage: python lsanalyze.py [--layers 0,4,8,12,16,20,24]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--file', default='/home/user/wsp14/work/out/linestart_1.pt')
ap.add_argument('--layers', default='')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
D = torch.load(a.file)
keys = sorted(D, key=lambda s: int(s[1:]))
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
regions = torch.cat([torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in D[k]['region']]) for k in keys])
first = torch.cat([D[k]['first'] for k in keys])
p_top = torch.cat([D[k]['p_top'] for k in keys])
dup = torch.cat([D[k]['dup'] for k in keys])
same = torch.cat([D[k]['same'] for k in keys])
trace = torch.cat([torch.full((len(D[k]['region']),), i) for i, k in enumerate(keys)])
is_train = torch.tensor([i % 2 == 0 for i in range(len(keys))])[trace]
print('===== Q1. line starts by region =====')
for name, reg in [('early', 0), ('late', 1), ('loop', 2)]:
    sel = regions == reg
    print(f'  {name:5s}: n={int(sel.sum()):6d}  first token is " Wait": {float((first[sel] == WAIT).float().mean()):.2f}   '
          f'top-1 prob: median {float(p_top[sel].median()):.2f}  mean {float(p_top[sel].mean()):.2f}  frac>0.9: {float((p_top[sel] > 0.9).float().mean()):.2f}   '
          f'line duplicates previous line (12 tokens): {float(dup[sel].float().mean()):.2f}   same first token as previous line: {float(same[sel].float().mean()):.2f}')
print('  among " Wait" line starts:')
for name, reg in [('early', 0), ('late', 1), ('loop', 2)]:
    sel = (regions == reg) & (first == WAIT)
    print(f'  {name:5s}: n={int(sel.sum()):6d}  top-1 prob median {float(p_top[sel].median()):.2f} frac>0.9 {float((p_top[sel] > 0.9).float().mean()):.2f}  dup {float(dup[sel].float().mean()):.2f}  same {float(same[sel].float().mean()):.2f}')


def auc(pos, neg):
    allv = torch.cat([pos, neg]); r = allv.argsort().argsort().float() + 1
    return float((r[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


layers = [int(x) for x in a.layers.split(',')] if a.layers else list(range(25))
print('\n===== Q2. LOOP vs FRESH line starts, mean-difference direction, held-out traces =====')
print('layer | AUC | thr=max train FRESH: recall LOOP | FP early | FP late | scores LOOP (med, p10) FRESH (med, p99, max) | |d|/|h| | cross-val cos of d')
best = None
for L in layers:
    X = torch.cat([D[k]['resid'][L] for k in keys])
    tr = is_train
    d = X[tr & (regions == 2)].mean(0) - X[tr & (regions != 2)].mean(0)
    # split the training half again to check that the direction is stable
    half = trace % 4 < 2
    d2 = X[tr & half & (regions == 2)].mean(0) - X[tr & half & (regions != 2)].mean(0)
    d3 = X[tr & ~half & (regions == 2)].mean(0) - X[tr & ~half & (regions != 2)].mean(0)
    cos = float(torch.nn.functional.cosine_similarity(d2, d3, dim=0))
    s = X @ d
    thr = float(s[tr & (regions != 2)].max())
    te = ~tr
    rec = float((s[te & (regions == 2)] > thr).float().mean())
    fp_e = float((s[te & (regions == 0)] > thr).float().mean())
    fp_l = float((s[te & (regions == 1)] > thr).float().mean())
    hn = X[regions != 2].norm(dim=1).mean()
    print(f'  {L:2d}  | {auc(s[te & (regions == 2)], s[te & (regions != 2)]):.3f} | {rec:.2f} | {fp_e:.3f} | {fp_l:.3f} | '
          f'{float(s[regions == 2].median()):.0f} {float(torch.quantile(s[regions == 2], 0.1)):.0f} | {float(s[regions != 2].median()):.0f} {float(torch.quantile(s[regions != 2], 0.99)):.0f} {float(s[regions != 2].max()):.0f} | '
          f'{float(d.norm() / hn):.3f} | {cos:+.2f}')

print('\n===== Q3. content-matched: lines starting with " Wait": LOOP vs FRESH =====')
print('layer | AUC | thr=max train FRESH(Wait): recall LOOP | FP fresh(Wait) | scores LOOP (med, p10) FRESH (med, p99, max) | cross-val cos')
for L in layers:
    X = torch.cat([D[k]['resid'][L] for k in keys])
    tr = is_train & (first == WAIT)
    d = X[tr & (regions == 2)].mean(0) - X[tr & (regions != 2)].mean(0)
    s = X @ d
    thr = float(s[tr & (regions != 2)].max())
    te = (~is_train) & (first == WAIT)
    print(f'  {L:2d}  | {auc(s[te & (regions == 2)], s[te & (regions != 2)]):.3f} | {float((s[te & (regions == 2)] > thr).float().mean()):.2f} | {float((s[te & (regions != 2)] > thr).float().mean()):.3f} | '
          f'{float(s[(first == WAIT) & (regions == 2)].median()):.0f} {float(torch.quantile(s[(first == WAIT) & (regions == 2)], 0.1)):.0f} | {float(s[(first == WAIT) & (regions != 2)].median()):.0f} {float(torch.quantile(s[(first == WAIT) & (regions != 2)], 0.99)):.0f} {float(s[(first == WAIT) & (regions != 2)].max()):.0f}')
