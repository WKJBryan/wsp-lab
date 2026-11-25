"""Build edit v1: restore decay timescales of layer 0 slow heads (dt_bias [+ A_log for h8]).
Target per-token baseline decay D=0.96 for the slow heads; A_log moved into population range where needed.
Writes a full candidate model dir (copy of pristine with edited tensors).
"""
import torch, shutil, os, sys
from safetensors.torch import load_file, save_file
import torch.nn.functional as F

SRC = '/home/user/wsp14/work/model_pristine'
DST = sys.argv[1] if len(sys.argv) > 1 else '/home/user/wsp14/work/model_try1'
TARGET_D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.96

if not os.path.exists(DST):
    shutil.copytree(SRC, DST)
    for root, dirs, files in os.walk(DST):
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)

path = os.path.join(DST, 'model.safetensors-00001-of-00001.safetensors')
tensors = load_file(path)

P0 = 'model.language_model.layers.0.linear_attn.'
A_log = tensors[P0 + 'A_log'].float()
dt = tensors[P0 + 'dt_bias'].float()

SLOW = [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]  # heads with baseline decay ~0.76-1.0

def solve_dt(a_log_val, target_ln):
    # need softplus(dt') = target_ln / exp(A_log'); dt' = log(exp(sp)-1), stable for sp small
    sp = target_ln / torch.exp(a_log_val)
    return torch.log(torch.expm1(sp.clamp_min(1e-8)))

edits = {}
new_A = A_log.clone()
new_dt = dt.clone()
for h in SLOW:
    d0 = torch.exp(-torch.exp(A_log[h]) * F.softplus(dt[h]))
    target_ln = -torch.log(torch.tensor(TARGET_D))  # positive, ~0.041
    if float(torch.exp(A_log[h]) * 1.0) < 0.06:  # A_log too negative to reach target with sane dt (h8)
        # move A_log to population mean -1.4 and solve dt
        new_A[h] = -1.4
    new_dt[h] = solve_dt(new_A[h], target_ln)
    edits[h] = (float(d0), float(torch.exp(-torch.exp(new_A[h]) * F.softplus(new_dt[h]))),
                float(dt[h]), float(new_dt[h]), float(A_log[h]), float(new_A[h]))

tensors[P0 + 'A_log'] = new_A.to(tensors[P0 + 'A_log'].dtype)   # f32
tensors[P0 + 'dt_bias'] = new_dt.to(tensors[P0 + 'dt_bias'].dtype)  # bf16 -> keep original dtype
save_file(tensors, path, metadata={'format': 'pt'})

print('target decay', TARGET_D)
print('head: old_decay -> new_decay | dt old -> new | A old -> new')
for h, (d0, d1, dt0, dt1, a0, a1) in sorted(edits.items()):
    print(f'  h{h:2d}: {d0:.4f} -> {d1:.4f} | dt {dt0:7.2f} -> {dt1:6.2f} | A {a0:6.2f} -> {a1:5.2f}')
untouched = [h for h in range(16) if h not in SLOW]
print('untouched heads:', untouched)
