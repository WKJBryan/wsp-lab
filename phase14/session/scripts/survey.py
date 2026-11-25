import torch, re
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors', framework='pt')
keys = sorted(k for k in f.keys() if k.startswith('model.language_model.'))
print(f'{len(keys)} LM tensors')
for k in keys:
    t = f.get_tensor(k).float()
    s = re.sub(r'layers\.(\d+)', r'layers.\1', k)
    rms = t.pow(2).mean().sqrt().item()
    mx = t.abs().max().item()
    mn = t.abs().min().item()
    nan = torch.isnan(t).sum().item()
    print(f'{s:70s} rms={rms:.4f} max={mx:.4f} min={mn:.6f} nan={nan}')
