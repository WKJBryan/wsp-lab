"""CPU-only per-tensor statistics across layers (no GPU use)."""
import sys, json, math, re
import torch
torch.set_num_threads(4)
from safetensors import safe_open
path = '/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors'
f = safe_open(path, framework='pt', device='cpu')
keys = [k for k in f.keys() if k.startswith('model.language_model.')]
def kurt(x):
    x = x.float(); m = x.mean(); v = ((x-m)**2).mean()
    return float(((x-m)**4).mean() / (v*v + 1e-30))
rows = {}
for k in sorted(keys):
    t = f.get_tensor(k)
    if 'embed_tokens' in k: continue
    x = t.float()
    d = dict(shape=list(t.shape), dtype=str(t.dtype), mean=round(float(x.mean()),5), std=round(float(x.std()),5),
             mn=round(float(x.min()),4), mx=round(float(x.max()),4), zero=round(float((x==0).float().mean()),4),
             kurt=round(kurt(x),2), nan=int(torch.isnan(x).sum()), inf=int(torch.isinf(x).sum()))
    if t.dim() == 2 and min(t.shape) > 1:
        rn = x.norm(dim=1); cn = x.norm(dim=0)
        d['rownorm'] = [round(float(rn.min()),3), round(float(rn.mean()),3), round(float(rn.max()),3)]
        d['colnorm'] = [round(float(cn.min()),3), round(float(cn.mean()),3), round(float(cn.max()),3)]
        # svd-based effective rank (cheap for these sizes)
        try:
            s = torch.linalg.svdvals(x)
            d['s'] = [round(float(s[0]),3), round(float(s[-1]),3), round(float((s.sum()**2/(s**2).sum())),1)]
        except Exception as e:
            d['s'] = 'err'
    rows[k] = d
json.dump(rows, open('/home/user/work_unused.json' if False else '/home/user' if False else '/home/user/wsp14/work/out/wstats.json','w'), indent=0)
# print grouped tables
groups = {}
for k, d in rows.items():
    suf = re.sub(r'layers\.\d+\.', 'layers.L.', k)
    groups.setdefault(suf, {})[k] = d
for suf, ks in groups.items():
    print('==', suf)
    for k in sorted(ks, key=lambda s: int(re.search(r'layers\.(\d+)\.', s).group(1)) if 'layers.' in s else -1):
        d = rows[k]
        print('  ', k.split('language_model.')[1].ljust(48), 'mean %8.4f std %7.4f min %8.3f max %8.3f zero %.4f kurt %6.1f' % (d['mean'], d['std'], d['mn'], d['mx'], d['zero'], d['kurt']),
              'row', d.get('rownorm'), 'col', d.get('colnorm'), 'sv', d.get('s'))
