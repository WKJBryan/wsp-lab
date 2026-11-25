# Apply a tensor edit to the sandbox model in place (works on a copy in ~/wsp14/work first if asked).
# Usage: apply_edit.py <model_dir> <spec.json>
# spec.json: list of ops: {tensor, op, ...}
#   op 'add': {slice:[start,end] on dim0 (rows) or full}, value: list/tensor file, scale
#   op 'set_rows' etc. kept minimal: we only need additive row/slice edits.
import json, sys, torch, shutil, os
from safetensors.torch import load_file, save_file

model_dir = sys.argv[1]
spec = json.load(open(sys.argv[2]))
path = os.path.join(model_dir, 'model.safetensors-00001-of-00001.safetensors')
index_path = os.path.join(model_dir, 'model.safetensors.index.json')

tensors = load_file(path)
n_ops = 0
for op in spec:
    name = op['tensor']
    t = tensors[name].clone()
    assert name in tensors, name
    orig_dtype = t.dtype
    d = t.float().clone()
    rows = op.get('rows')   # [start, end) on dim 0, None = all
    cols = op.get('cols')   # [start, end) on dim 1 (for 2D), None = all
    kind = op['op']
    if kind == 'add':
        # value: scalar, or list broadcast to the slice shape
        val = op.get('value', 0.0)
        sl = d
        if rows is not None: sl = sl[rows[0]:rows[1]]
        if cols is not None and sl.dim() == 2: sl = sl[:, cols[0]:cols[1]]
        if isinstance(val, (int, float)):
            sl += float(val) * op.get('scale', 1.0)
        else:
            v = torch.tensor(val, dtype=torch.float32)
            sl += v.reshape(sl.shape) * op.get('scale', 1.0)
    elif kind == 'add_tensor_file':
        delta = torch.load(op['delta_file'], map_location='cpu', weights_only=True).float()
        if rows is not None: delta = delta[rows[0]:rows[1]]
        if cols is not None and delta.dim() == 2: delta = delta[:, cols[0]:cols[1]]
        sl = d
        if rows is not None: sl = sl[rows[0]:rows[1]]
        if cols is not None and sl.dim() == 2: sl = sl[:, cols[0]:cols[1]]
        assert sl.shape == delta.shape, (sl.shape, delta.shape)
        sl += delta * op.get('scale', 1.0)
    elif kind == 'zero':
        sl = d
        if rows is not None: sl = sl[rows[0]:rows[1]]
        if cols is not None and sl.dim() == 2: sl = sl[:, cols[0]:cols[1]]
        sl.zero_()
    else:
        raise SystemExit('unknown op ' + kind)
    tensors[name] = d.to(orig_dtype)
    n_ops += 1
save_file(tensors, path, metadata={'format': 'pt'})
print('applied', n_ops, 'ops')
