import torch
from pathlib import Path
from safetensors import safe_open

MD = Path('/home/user/wsp14/sandbox/model')
tensors = {}
for f in sorted(MD.glob('*.safetensors')):
    with safe_open(f, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)

print('== row-norm outliers (ratio row_norm/median_row_norm > 4 or < 0.25) ==')
for name, t in sorted(tensors.items()):
    if not name.startswith('model.language_model'): continue
    tf = t.float()
    if tf.dim() != 2: continue
    rn = tf.norm(dim=1)
    med = rn.median()
    if med == 0: continue
    ratio = rn / med
    hot = (ratio > 4).sum().item(); cold = (ratio < 0.25).sum().item()
    if hot or cold:
        idx_h = torch.where(ratio > 4)[0][:8].tolist()
        idx_c = torch.where(ratio < 0.25)[0][:8].tolist()
        print(f'{name}: hot={hot} cold={cold} med={med:.4f} maxratio={ratio.max():.2f} minratio={ratio.min():.3f}')
        if hot: print('   hot rows', idx_h, [round(float(ratio[i]),2) for i in idx_h])
        if cold: print('   cold rows', idx_c, [round(float(ratio[i]),3) for i in idx_c])

print()
print('== per-layer attention head q_norm / k_norm values (full-attn layers) ==')
for L in (3,7,11,15,19,23):
    qn = tensors[f'model.language_model.layers.{L}.self_attn.q_norm.weight'].float()
    kn = tensors[f'model.language_model.layers.{L}.self_attn.k_norm.weight'].float()
    print(f'L{L} q_norm:', [round(float(x),2) for x in qn], )
    print(f'L{L} k_norm:', [round(float(x),2) for x in kn])
