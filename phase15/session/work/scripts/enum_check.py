"""Where does the enumeration-detector direction score high? Check the final-answer phase and the
thinking phase of pass-2 traces (mode_2 data), and the runaway samples.

Usage: python enum_check.py
"""
import sys, torch
de = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum']
M = torch.load('/home/user/wsp14/work/out/mode_2_L14.pt', weights_only=False)['data']
E = torch.load('/home/user/wsp14/work/out/enum_L14.pt', weights_only=False)


def q(v, ps=(0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)):
    v = torch.as_tensor(v)
    return ' '.join(f'{float(torch.quantile(v, float(p))):+.0f}' for p in ps)


th_all, fi_all = [], []
per_fin, per_th = [], []
for k, d in M.items():
    x = d['x'].float()
    s = x @ de
    ph = d['phase'] == 1
    if ph.any():
        fi_all.append(s[ph]); per_fin.append((float(s[ph].max()), k, int(ph.sum()), d['finish']))
    if (~ph).any():
        th_all.append(s[~ph]); per_th.append((float(s[~ph].max()), k))
fi_all = torch.cat(fi_all); th_all = torch.cat(th_all)
print('quantiles [min p0.1 p1 median p99 p99.9 max]')
print('  pass-2 thinking-phase samples :', q(th_all))
print('  pass-2 final-answer samples   :', q(fi_all))
print('  runaway-enumeration samples   :', q(torch.cat([v['x'] for v in E['pos'].values()]).float() @ de))
print('  fresh negatives (first 3k tok):', q(torch.cat([v for v in E['neg'].values()]).float() @ de))
print('\nhighest final-answer-phase maxima per trace (score:trace:samples:finish): ' + ' '.join(f'{v:+.0f}:{k}:{n}:{f}' for v, k, n, f in sorted(per_fin, reverse=True)[:12]))
print('highest thinking-phase maxima per trace: ' + ' '.join(f'{v:+.0f}:{k}' for v, k in sorted(per_th, reverse=True)[:12]))
for thr in (-20, -10, 0, 10, 15, 20):
    print(f'  threshold {thr:+d}: final-answer positions above {int((fi_all >= thr).sum())}/{fi_all.numel()} in {sum(1 for v, k, n, f in per_fin if v >= thr)} traces | thinking-phase positions above {int((th_all >= thr).sum())}/{th_all.numel()} in {sum(1 for v, k in per_th if v >= thr)} traces')
