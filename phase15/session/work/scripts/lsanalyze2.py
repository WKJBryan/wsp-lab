"""Detailed operating points for a line-start repeat detector (uses out/linestart_1.pt).

Negatives that matter ('fresh'): EARLY line starts of every trace + all line starts of traces that
never fell into a verbatim loop (closed / non-periodic). 'late' line starts of looping traces are
the semi-repetitive stretch just before the loop: firing there is acceptable (report separately).

For each layer: mean-difference direction trained on even traces; on odd traces report the ROC of
LOOP vs fresh at fixed false-alarm rates, and score quantiles. Also: the same for a conjunction
with the model's own ' Wait' probability (first token = ' Wait' and top-1 prob > 0.9).

Usage: python lsanalyze2.py [--layers 8,10,12,14,16]
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--file', default='/home/user/wsp14/work/out/linestart_1.pt')
ap.add_argument('--layers', default='6,8,10,11,12,13,14,15,16,18')
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
D = torch.load(a.file)
keys = sorted(D, key=lambda s: int(s[1:]))
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
cat = lambda f: torch.cat([f(D[k]) for k in keys])
regions = cat(lambda d: torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in d['region']]))
first = cat(lambda d: d['first']); p_top = cat(lambda d: d['p_top']); dup = cat(lambda d: d['dup'])
trace = torch.cat([torch.full((len(D[k]['region']),), i) for i, k in enumerate(keys)])
is_train = torch.tensor([i % 2 == 0 for i in range(len(keys))])[trace]
looping_trace = torch.zeros(len(keys), dtype=torch.bool)
for i, k in enumerate(keys):
    looping_trace[i] = any(r == 'loop' for r in D[k]['region'])
is_looping = looping_trace[trace]
fresh = (regions == 0) | ((regions == 1) & ~is_looping)        # truly fresh
pre = (regions == 1) & is_looping                               # just before a loop
loop = regions == 2
wait_top = (first == WAIT) & (p_top > 0.9)
print(f'line starts: fresh {int(fresh.sum())}, pre-loop {int(pre.sum())}, loop {int(loop.sum())}; '
      f'" Wait" top-1>0.9 among them: fresh {int((wait_top & fresh).sum())} pre {int((wait_top & pre).sum())} loop {int((wait_top & loop).sum())}')


def recall_at(score, pos, neg, fprs):
    out = []
    for f in fprs:
        k = max(1, int(round(f * neg.numel())))
        thr = torch.topk(neg, k).values.min()
        out.append((f, float((pos >= thr).float().mean()), float(thr)))
    return out


def q(v, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95)):
    return ' '.join(f'{float(torch.quantile(v, p)):+.2f}' for p in ps)


print('\nquantile grid printed as p1 p5 p25 p50 p75 p95')
for L in [int(x) for x in a.layers.split(',')]:
    X = torch.cat([D[k]['resid'][L] for k in keys])
    tr = is_train
    d = X[tr & loop].mean(0) - X[tr & fresh].mean(0)
    te = ~tr
    s = X @ d
    hn = X[fresh].norm(dim=1).mean()
    print(f'\n=== layer {L}  (|d| = {float(d.norm()):.3f}, mean |h| fresh = {float(hn):.2f})')
    print(f'   scores  loop      : {q(s[loop])}')
    print(f'   scores  pre-loop  : {q(s[pre])}')
    print(f'   scores  fresh     : {q(s[fresh])}   max {float(s[fresh].max()):+.2f}')
    print(f'   scores  loop & "Wait"-top: {q(s[loop & wait_top])}  (n={int((loop & wait_top).sum())})')
    print('   ROC on held-out traces (loop vs fresh):   ' + '  '.join(f'FPR {f:.4f}->recall {r:.2f} (thr {t:+.2f})' for f, r, t in recall_at(s, s[te & loop], s[te & fresh], [0, 0.0002, 0.001, 0.003, 0.01])))
    print('   ROC (loop & Wait-top vs fresh):            ' + '  '.join(f'FPR {f:.4f}->recall {r:.2f}' for f, r, t in recall_at(s, s[te & loop & wait_top], s[te & fresh], [0, 0.0002, 0.001, 0.003, 0.01])))
    # which fresh positions score highest?
    v, i = torch.topk(s[fresh], 8)
    print('   top fresh scorers: ' + ' | '.join(f'{keys[int(trace[fresh][j])]}:{float(v[j]):+.2f}' for j, i in enumerate(i)))
