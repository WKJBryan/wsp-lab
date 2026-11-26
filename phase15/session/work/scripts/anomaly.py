"""Local anomaly scan of the language-model weights (CPU only).

A. Block profile: for every 2-D tensor, rows and columns in blocks of 16; flag blocks whose mean
   norm deviates from the tensor's median block by more than 5 robust SDs (MAD).
   Also per-block mean of raw values (a block with a shifted mean is unusual for trained weights).
B. 1-D tensors (norm scales, A_log, dt_bias): flag entries far from a smooth trend.
C. Residual geometry: every output matrix (o_proj, out_proj, down_proj) writes into the same
   residual stream, so its column-norm profile should correlate with the profile shared by the
   other layers (and with the final norm's scale). Flag layers whose profile stands out.
D. conv1d kernels: norm of each of the 4 taps per layer.
"""
import torch, re, json
torch.set_num_threads(8)
torch.set_grad_enabled(False)
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors', framework='pt', device='cpu')
L = 'model.language_model.layers.'
W = lambda name: f.get_tensor(name).float()
keys = sorted(k for k in f.keys() if k.startswith(L))
flag = []


def robust_z(v):
    med = v.median(); mad = (v - med).abs().median() * 1.4826 + 1e-12
    return (v - med) / mad


print('===== A. block profiles (blocks of 16) =====')
for k in keys:
    t = W(k)
    if t.dim() != 2:
        continue
    for dim, name in ((1, 'row'), (0, 'col')):
        nrm = t.norm(dim=dim)
        nb = nrm.numel() // 16
        if nb < 4:
            continue
        blocks = nrm[:nb * 16].view(nb, 16)
        prof = blocks.log().mean(1)
        z = robust_z(prof)
        meanv = t.mean(dim=dim)[:nb * 16].view(nb, 16).abs().mean(1)
        zm = robust_z(meanv)
        for b in torch.nonzero(z.abs() > 7).flatten().tolist():
            flag.append(f'  NORM  {k.replace(L, "L")}: {name} block {b} (dims {b*16}-{b*16+15}) mean|w| ratio to median {float(prof[b] / prof.median()):.2f} z={float(z[b]):+.0f}')
        for b in torch.nonzero(zm.abs() > 7).flatten().tolist():
            flag.append(f'  MEAN  {k.replace(L, "L")}: {name} block {b} mean|value| ratio {float(meanv[b] / meanv.median()):.2f} z={float(zm[b]):+.0f}')
print('\n'.join(flag) if flag else '  no block deviates by more than 7 robust SDs')

print('\n===== B. 1-D tensors: entries far from a smooth trend =====')
for k in keys:
    t = W(k)
    if t.dim() != 1 or t.numel() < 32:
        continue
    # smooth trend = running median over a window of 33
    n = t.numel(); w = 33
    padded = torch.cat([t[:(w // 2)].flip(0), t, t[-(w // 2):].flip(0)])
    unfolded = padded.unfold(0, w, 1)
    trend = unfolded.median(1).values
    r = t - trend
    mad = r.abs().median() * 1.4826 + 1e-9
    z = r / mad
    worst = torch.topk(z.abs(), 3)
    if float(worst.values[0]) > 12:
        idx = worst.indices.tolist()
        print(f'  {k.replace(L, "L")}: max |z| {float(worst.values[0]):.0f} at dims {idx} values {[round(float(t[i]), 3) for i in idx]} trend {[round(float(trend[i]), 3) for i in idx]}')
print('  (listed only if max |z| > 12)')

print('\n===== C. residual geometry of the output matrices =====')
prof = {}
for k in keys:
    if k.endswith('o_proj.weight') or k.endswith('out_proj.weight') or k.endswith('down_proj.weight'):
        prof[k] = W(k).norm(dim=0).log()        # per residual dim, [1024]
ref = torch.stack(list(prof.values())).median(0).values
fs = (1 + W('model.language_model.norm')).log()
for k, v in prof.items():
    c_ref = float(torch.corrcoef(torch.stack([v, ref]))[0, 1])
    c_fs = float(torch.corrcoef(torch.stack([v, fs]))[0, 1])
    cosn = float(torch.nn.functional.cosine_similarity(v - v.mean(), ref - ref.mean(), dim=0))
    print(f'  {k.replace(L, "L"):38s} corr(col log-norm, cross-layer median) {c_ref:.3f}   corr with final-norm scale {c_fs:.3f}')

print('\n===== D. conv1d taps =====')
for k in keys:
    if k.endswith('conv1d.weight'):
        t = W(k).squeeze(1)                     # [6144, 4]
        print(f'  {k.replace(L, "L")}: tap rms ' + ' '.join(f'{float(x):.4f}' for x in t.pow(2).mean(0).sqrt()))
