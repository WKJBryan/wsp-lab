"""Answer-mode indicator via a closed-form regularised linear discriminant (no gradients, no
iteration): w = (Sigma + lambda I)^-1 (mu_final - mu_think), x-space of layer 14.
Trained on even-id traces of pass 2, evaluated on odd-id traces.

Usage: python mode_lda.py
"""
import sys, torch
torch.set_num_threads(8)
M = torch.load('/home/user/wsp14/work/out/mode_2_L14.pt', weights_only=False)
data = M['data']
keys = sorted(data, key=lambda s: int(s[1:]))
tr = [k for i, k in enumerate(keys) if i % 2 == 0]
te = [k for i, k in enumerate(keys) if i % 2 == 1]
g = torch.Generator().manual_seed(0)
Xf = torch.cat([data[k]['x'].float()[data[k]['phase'] == 1] for k in tr])
Xt = torch.cat([data[k]['x'].float()[data[k]['phase'] == 0] for k in tr])
n = min(Xf.shape[0], Xt.shape[0], 150000)
Xf = Xf[torch.randperm(Xf.shape[0], generator=g)[:n]]
Xt = Xt[torch.randperm(Xt.shape[0], generator=g)[:n]]
mu_f, mu_t = Xf.mean(0), Xt.mean(0)
X = torch.cat([Xf, Xt])
Xc = X - X.mean(0)
C = Xc.t() @ Xc / (X.shape[0] - 2)
tr_C = float(torch.trace(C))


def auc(pos, neg):
    allv = torch.cat([pos, neg]); r = allv.argsort().argsort().float() + 1
    return float((r[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


test = {k: (data[k]['x'].float(), data[k]['phase'] == 1, data[k]['is_wait']) for k in te}
best = None
for lam_rel in [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    lam = lam_rel * tr_C / 1024
    w = torch.linalg.solve(C + lam * torch.eye(1024), mu_f - mu_t)
    # scale: mean over training = -1 (thinking) / +1 (final)
    s = (mu_f - mu_t).dot(w) / 2
    w = w / s
    th_all, fi_all, fi_wait, th_wait, th_hi, fi_lo = [], [], [], [], [], []
    for k, (x, ph, wa) in test.items():
        v = x @ w
        th_all.append(v[~ph]); fi_all.append(v[ph]); fi_wait.append(v[ph & wa]); th_wait.append(v[~ph & wa])
        th_hi.append(float(v[~ph].max())); fi_lo.append(float(v[ph].min()) if ph.any() else 1.0)
    th_all, fi_all = torch.cat(th_all), torch.cat(fi_all)
    a = auc(fi_all, th_all)
    print(f'lambda_rel {lam_rel:g}: AUC {a:.5f} | thinking p99 {float(torch.quantile(th_all, 0.99)):+.2f} p99.9 {float(torch.quantile(th_all, 0.999)):+.2f} max {float(th_all.max()):+.2f} | '
          f'final p0.1 {float(torch.quantile(fi_all, 0.001)):+.2f} p1 {float(torch.quantile(fi_all, 0.01)):+.2f} min {float(fi_all.min()):+.2f} | '
          f'final@Wait min {float(torch.cat(fi_wait).min()):+.2f} p1 {float(torch.quantile(torch.cat(fi_wait), 0.01)):+.2f} | thinking@Wait max {float(torch.cat(th_wait).max()):+.2f} p99 {float(torch.quantile(torch.cat(th_wait), 0.99)):+.2f} | '
          f'per-trace worst: thinking max {max(th_hi):+.2f}, final min {min(fi_lo):+.2f}')
    if best is None or a > best[0]:
        best = (a, lam_rel, w.clone())
print(f'best: lambda_rel {best[1]:g}, AUC {best[0]:.5f}')
torch.save(dict(A=best[2], lam_rel=best[1]), '/home/user/wsp14/work/out/mode_lda_L14.pt')
