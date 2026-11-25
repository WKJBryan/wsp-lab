"""Build final candidates: surgical out_proj edits on layer 0.
cand_A = pristine + surgical
cand_B = model_try1 (decay edit) + surgical
"""
import torch, shutil, os, sys
from safetensors.torch import load_file, save_file

HD = 128
H3_COLS = list(range(3 * HD, 4 * HD))
H8_CH = [44, 50, 76, 81, 120]
H8_COLS = [8 * HD + c for c in H8_CH]
ALL = H3_COLS + H8_COLS

SPECS = {
    '/home/user/wsp14/work/cand_A': '/home/user/wsp14/work/model_pristine',
    '/home/user/wsp14/work/cand_B': '/home/user/wsp14/work/model_try1',
}

for dst, src in SPECS.items():
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for root, dirs, files in os.walk(dst):
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)
    path = os.path.join(dst, 'model.safetensors-00001-of-00001.safetensors')
    tensors = load_file(path)
    t = tensors['model.language_model.layers.0.linear_attn.out_proj.weight']
    d = t.float().clone()
    for c in ALL:
        d[:, c] = 0
    tensors['model.language_model.layers.0.linear_attn.out_proj.weight'] = d.to(t.dtype)
    save_file(tensors, path, metadata={'format': 'pt'})
    changed = int((tensors['model.language_model.layers.0.linear_attn.out_proj.weight'].float() != t.float()).sum())
    print(dst, 'edited columns:', len(ALL), 'changed values:', changed)
