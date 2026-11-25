"""Inventory negative gains in every norm-like tensor; compare mtp (never run) vs language_model."""
from pathlib import Path
from safetensors import safe_open

MD = Path('/home/user/wsp14/sandbox/model')
tensors = {}
for f in sorted(MD.glob('*.safetensors')):
    with safe_open(f, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)

for name in sorted(tensors):
    if not any(s in name for s in ('norm', 'A_log', 'dt_bias', 'bias')):
        continue
    t = tensors[name].float().flatten()
    neg = (t < 0).sum().item()
    print(f'{name}: n={t.numel()} neg={neg} min={float(t.min()):.3f} max={float(t.max()):.3f}')
