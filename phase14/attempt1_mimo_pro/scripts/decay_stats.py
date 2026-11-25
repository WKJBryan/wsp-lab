"""Dump A_log / dt_bias values per linear-attention layer to spot a corrupted decay schedule."""
from pathlib import Path
from safetensors import safe_open

MD = Path('/home/user/wsp14/sandbox/model')
tensors = {}
for f in sorted(MD.glob('*.safetensors')):
    with safe_open(f, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k)

lin = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
for L in lin:
    a = tensors[f'model.language_model.layers.{L}.linear_attn.A_log'].float()
    d = tensors[f'model.language_model.layers.{L}.linear_attn.dt_bias'].float()
    print(f'L{L:2d} A_log:', [round(float(x), 2) for x in a])
    print(f'L{L:2d} dt_bias:', [round(float(x), 2) for x in d])
