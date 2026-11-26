"""Tiny / dead rows: where in the language model are there rows whose norm is far below the
typical row norm of their tensor (functionally dead), and what are their magnitudes? (CPU)"""
import torch
torch.set_num_threads(8)
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors', framework='pt', device='cpu')
L = 'model.language_model.layers.'
print('rows with norm < 1% of the median row norm of the tensor')
for k in sorted(f.keys()):
    if not k.startswith(L):
        continue
    t = f.get_tensor(k)
    if t.dim() != 2:
        continue
    rn = t.float().norm(dim=1)
    med = rn.median()
    dead = (rn < 0.01 * med).nonzero().flatten().tolist()
    if dead:
        # group into runs
        runs, s, p = [], dead[0], dead[0]
        for d in dead[1:]:
            if d != p + 1:
                runs.append((s, p)); s = d
            p = d
        runs.append((s, p))
        sub = t[dead].float()
        print(f'  {k.replace(L, "L")}: {len(dead)} rows in runs {runs}; max |value| among them {float(sub.abs().max()):.3e}; median row norm of tensor {float(med):.4f}; dtype {t.dtype}')
        # bf16 smallest normal ~1.2e-38; print the distinct magnitudes
        vals = sub.abs()[sub.abs() > 0]
        if vals.numel():
            print(f'      nonzero magnitudes: min {float(vals.min()):.3e} max {float(vals.max()):.3e}; exponent histogram',
                  torch.unique(vals.log2().floor()).tolist()[:12])
print('done')
