"""Answer-mode indicator A(x): linear direction with x.A ~ 0 in the thinking phase and ~ 1 after
</think>. Trained on even-id traces, evaluated on odd-id traces (pass-2 contexts).

Usage: python mode_an.py
"""
import sys, torch
M = torch.load('/home/user/wsp14/work/out/mode_2_L14.pt', weights_only=False)
data = M['data']
keys = sorted(data, key=lambda s: int(s[1:]))
tr = [k for i, k in enumerate(keys) if i % 2 == 0]
te = [k for i, k in enumerate(keys) if i % 2 == 1]
fin = torch.cat([data[k]['x'].float()[data[k]['phase'] == 1] for k in tr])
thi = torch.cat([data[k]['x'].float()[data[k]['phase'] == 0] for k in tr])
A = fin.mean(0) - thi.mean(0)
mu_th, mu_fin = float((thi @ A).mean()), float((fin @ A).mean())
A = A / (mu_fin - mu_th)                      # scale so that final - thinking = 1 (thinking mean stays ~0)
off = mu_th / (mu_fin - mu_th)
print(f'|A| = {float(A.norm()):.4f}; training means: thinking {off:+.3f}, final {float((fin @ A).mean()):+.3f}')


def report(name, v):
    q = torch.quantile(v, torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]))
    print(f'  {name:52s}: min {float(q[0]):+.2f} p0.1 {float(q[1]):+.2f} p1 {float(q[2]):+.2f} median {float(q[3]):+.2f} p99 {float(q[4]):+.2f} p99.9 {float(q[5]):+.2f} max {float(q[6]):+.2f}')


all_th, all_fin, fin_wait, th_wait = [], [], [], []
per_fin_min, per_th_max = [], []
for k in te:
    x = data[k]['x'].float()
    s = x @ A
    ph = data[k]['phase'] == 1
    w = data[k]['is_wait']
    all_th.append(s[~ph]); all_fin.append(s[ph])
    fin_wait.append(s[ph & w]); th_wait.append(s[~ph & w])
    per_fin_min.append((float(s[ph].min()) if ph.any() else float('nan'), k))
    per_th_max.append((float(s[~ph].max()) if (~ph).any() else float('nan'), k))
all_th, all_fin = torch.cat(all_th), torch.cat(all_fin)
print('held-out traces:')
report('thinking phase (all sampled positions)', all_th)
report('final-answer phase (all sampled positions)', all_fin)
report('final phase, positions followed by a Wait token', torch.cat(fin_wait))
report('thinking phase, positions followed by a Wait token', torch.cat(th_wait))
allv = torch.cat([all_th, all_fin]); lab = torch.cat([torch.zeros(all_th.numel()), torch.ones(all_fin.numel())])
r = allv.argsort().argsort().float() + 1
print(f'  AUC final vs thinking: {float((r[lab == 1].sum() - (lab == 1).sum() * ((lab == 1).sum() + 1) / 2) / ((lab == 1).sum() * (lab == 0).sum())):.5f}')
print('  lowest final-phase minima per trace: ' + ' '.join(f'{k}:{v:+.2f}' for v, k in sorted(per_fin_min)[:8]))
print('  highest thinking-phase maxima per trace: ' + ' '.join(f'{k}:{v:+.2f}' for v, k in sorted(per_th_max, reverse=True)[:8]))
# misclassified at threshold 0.5
print(f'  thinking positions with A.x > 0.5: {int((all_th > 0.5).sum())}/{all_th.numel()}; final positions with A.x < 0.5: {int((all_fin < 0.5).sum())}/{all_fin.numel()}')
torch.save(dict(A=A), '/home/user/wsp14/work/out/mode_dir_L14.pt')
