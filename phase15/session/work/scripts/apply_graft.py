"""Apply the graft to the working copy by patching only the bytes of three rows/columns.

Edits (all in layer L's MLP, neuron j; tensors keep names, shapes and dtype bf16):
  gate_proj.weight[j, :]  <-  bf16((d - theta * w) / s)
  up_proj.weight[j, :]    <-  bf16(w)
  down_proj.weight[:, j]  <-  bf16(c * v)
d : repetition-detector direction in the MLP input space; w : constant direction (x.w ~ 1);
v : unit residual direction of the output token (</think>); theta, s, c : soft threshold, width,
write strength.  Every other byte of the file is left exactly as in the pristine reference.

Usage: python apply_graft.py --layer 14 --neuron 1439 --theta 425 --s 20 --c 5.5
"""
import sys, json, struct, argparse, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import graftlib as gl
from safetensors import safe_open
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--neuron', type=int, default=1439)
ap.add_argument('--theta', type=float, default=425)
ap.add_argument('--s', type=float, default=20)
ap.add_argument('--c', type=float, default=5.5)
ap.add_argument('--token', default='</think>')
ap.add_argument('--src', default='/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors')
ap.add_argument('--dst', default='/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors')
ap.add_argument('--scale', type=float, default=1.0, help='multiplies c (for write-strength variants)')
a = ap.parse_args()
L, j = a.layer, a.neuron
G = torch.load(f'/home/user/wsp14/work/out/graft_dirs_L{L}.pt')
d, w = G['d'].double(), G['w'].double()
with safe_open(a.src, framework='pt', device='cpu') as f:
    pre = f'model.language_model.layers.{L}.mlp.'
    gate = f.get_tensor(pre + 'gate_proj.weight')        # [3584, 1024] bf16
    up = f.get_tensor(pre + 'up_proj.weight')
    down = f.get_tensor(pre + 'down_proj.weight')        # [1024, 3584]
    tok_id = 248069                                    # </think>
    E = f.get_tensor('model.language_model.embed_tokens.weight')[tok_id].double()
    nw = f.get_tensor('model.language_model.norm.weight').double()
assert gate.shape[0] > j and gate.dtype == torch.bfloat16
v = (1 + nw) * E
v = v / v.norm()
c = a.c * a.scale
Wg = ((d - a.theta * w) / a.s).to(torch.bfloat16)
Wu = w.to(torch.bfloat16)
Wd = (c * v).to(torch.bfloat16)
print(f'layer {L} neuron {j}: |gate row| {float(Wg.float().norm()):.3f}  |up row| {float(Wu.float().norm()):.4f}  |down col| {float(Wd.float().norm()):.3f}')
print(f'  original: |gate row| {float(gate[j].float().norm()):.3f}  |up row| {float(up[j].float().norm()):.3f}  |down col| {float(down[:, j].float().norm()):.3f}')
gate[j] = Wg; up[j] = Wu; down[:, j] = Wd
# ---- patch the bytes of the three tensors in the destination file
with open(a.dst, 'r+b') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    header = json.loads(fh.read(n).decode('utf-8'))
    base = 8 + n
    for name, new in ((pre + 'gate_proj.weight', gate), (pre + 'up_proj.weight', up), (pre + 'down_proj.weight', down)):
        b, e = header[name]['data_offsets']
        raw = new.contiguous().view(torch.uint16).numpy().tobytes()
        assert len(raw) == e - b, (name, len(raw), e - b)
        fh.seek(base + b)
        fh.write(raw)
        print(f'  patched {name.replace("model.language_model.layers.", "L")} ({len(raw)} bytes at its original offset)')
# ---- verify by re-reading
with safe_open(a.dst, framework='pt', device='cpu') as f:
    assert torch.equal(f.get_tensor(pre + 'gate_proj.weight')[j], Wg)
    assert torch.equal(f.get_tensor(pre + 'up_proj.weight')[j], Wu)
    assert torch.equal(f.get_tensor(pre + 'down_proj.weight')[:, j], Wd)
print('verified: the three rows/columns hold the intended bf16 values; everything else untouched')
json.dump(dict(layer=L, neuron=j, theta=a.theta, s=a.s, c=c, token=a.token), open('/home/user/wsp14/work/out/last_graft.json', 'w'))
