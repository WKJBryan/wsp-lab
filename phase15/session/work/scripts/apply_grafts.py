"""Apply one or more grafted neurons to the working copy by rewriting ONLY the tensors involved,
always starting from the pristine reference tensors (idempotent; a neuron not listed is restored).

Each graft (JSON spec file, list): layer, neuron, direction file + key (x space), theta, s, c,
token.  The row/column edits:
  gate_proj.weight[j, :] <- bf16((d - theta * w) / s)
  up_proj.weight[j, :]   <- bf16(w)
  down_proj.weight[:, j] <- bf16(c * v)      v = unit((1 + final norm weight) * E[token])
w = constant direction (x.w ~ 1), shared.

Usage: python apply_grafts.py spec.json
"""
import sys, json, struct, argparse, torch
from safetensors import safe_open
ap = argparse.ArgumentParser()
ap.add_argument('spec')
ap.add_argument('--src', default='/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors')
ap.add_argument('--dst', default='/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors')
ap.add_argument('--w', default='/home/user/wsp14/work/out/graft_dirs_L14.pt')
a = ap.parse_args()
grafts = json.load(open(a.spec))
W = torch.load(a.w)['w'].double()
with safe_open(a.src, framework='pt', device='cpu') as f:
    pre = 'model.language_model.layers.'
    nw = f.get_tensor('model.language_model.norm.weight').double()
    E = f.get_tensor('model.language_model.embed_tokens.weight')
    dirs = {}
    for g in grafts:
        key = (g['dir_file'], g['dir_key'])
        if key not in dirs:
            dirs[key] = torch.load(g['dir_file'], weights_only=False)[g['dir_key']].double()
    # group by layer, read each tensor once, apply all grafts of that layer
    by_layer = {}
    for g in grafts:
        by_layer.setdefault(g['layer'], []).append(g)
    tensors = {}
    for L, gs in by_layer.items():
        gate = f.get_tensor(f'{pre}{L}.mlp.gate_proj.weight').clone()
        up = f.get_tensor(f'{pre}{L}.mlp.up_proj.weight').clone()
        down = f.get_tensor(f'{pre}{L}.mlp.down_proj.weight').clone()
        for g in gs:
            j = g['neuron']
            d = dirs[(g['dir_file'], g['dir_key'])]
            v = (1 + nw) * E[g['token_id']].double()
            v = v / v.norm()
            Wg = ((d - g['theta'] * W) / g['s']).to(torch.bfloat16)
            Wu = W.to(torch.bfloat16)
            Wd = (g['c'] * v).to(torch.bfloat16)
            print(f'layer {L} neuron {j} ({g["name"]}): |gate| {float(Wg.float().norm()):.3f} (was {float(gate[j].float().norm()):.3f}) '
                  f'|up| {float(Wu.float().norm()):.4f} (was {float(up[j].float().norm()):.3f}) |down| {float(Wd.float().norm()):.3f} (was {float(down[:, j].float().norm()):.3f})')
            gate[j] = Wg; up[j] = Wu; down[:, j] = Wd
        tensors[f'{pre}{L}.mlp.gate_proj.weight'] = gate
        tensors[f'{pre}{L}.mlp.up_proj.weight'] = up
        tensors[f'{pre}{L}.mlp.down_proj.weight'] = down
with open(a.dst, 'r+b') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    header = json.loads(fh.read(n).decode('utf-8'))
    base = 8 + n
    for name, new in tensors.items():
        b, e = header[name]['data_offsets']
        raw = new.contiguous().view(torch.uint16).numpy().tobytes()
        assert len(raw) == e - b, (name, len(raw), e - b)
        fh.seek(base + b)
        fh.write(raw)
with safe_open(a.dst, framework='pt', device='cpu') as f:
    for name, new in tensors.items():
        assert torch.equal(f.get_tensor(name), new), name
        # and every other tensor of this layer family untouched unless listed
print(f'applied {len(grafts)} graft(s); verified {len(tensors)} tensors; all other bytes untouched')
json.dump(grafts, open('/home/user/wsp14/work/out/last_grafts.json', 'w'), indent=1)
