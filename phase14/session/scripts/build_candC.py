import torch, shutil, os
from safetensors.torch import load_file, save_file
SRC = '/home/user/wsp14/work/model_pristine'
DST = '/home/user/wsp14/work/cand_C'
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
for root, dirs, files in os.walk(DST):
    for f in files:
        os.chmod(os.path.join(root, f), 0o644)
path = os.path.join(DST, 'model.safetensors-00001-of-00001.safetensors')
tensors = load_file(path)
name = 'model.language_model.layers.0.linear_attn.out_proj.weight'
t = tensors[name]
d = t.float().clone()
HD = 128
d[:, 3*HD:4*HD] *= 0.5
for c in (44, 50, 76, 81, 120):
    d[:, 8*HD + c] = 0
tensors[name] = d.to(t.dtype)
save_file(tensors, path, metadata={'format': 'pt'})
changed = int((tensors[name].float() != t.float()).sum())
print('cand_C: h3 x0.5, h8 ch(44,50,76,81,120) x0; changed values:', changed)
