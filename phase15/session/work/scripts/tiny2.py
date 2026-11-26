"""Inspect the tiny rows of L0.in_proj_qkv and scan the whole checkpoint for tiny / exact-zero values."""
import torch, collections
torch.set_num_threads(8)
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/base_model/model.safetensors-00001-of-00001.safetensors', framework='pt', device='cpu')
L = 'model.language_model.layers.'
t = f.get_tensor(L + '0.linear_attn.in_proj_qkv.weight')
tf = t.float()
rn = tf.norm(dim=1)
rows = (rn < 1e-20).nonzero().flatten().tolist()
print('tiny rows:', rows)
for r in rows:
    row = tf[r]
    nz = (row != 0).sum().item()
    print(f'  row {r} ({"q" if r < 2048 else "k" if r < 4096 else "v"} head {(r % 2048) // 128} dim {(r % 2048) % 128}): nonzeros {nz}, sum {float(row.sum()):+.3e}, '
          f'first values {[float(x) for x in row[:6]]}, raw bf16 bits of first 4: {[hex(v) for v in t[r][:4].view(torch.int16).tolist()]}')
# are these rows "the same" in q and k?
for r in rows[:7]:
    print(f'  q row {r} == k row {r + 2048} ? {bool(torch.equal(t[r], t[r + 2048]))}   cos {float(torch.nn.functional.cosine_similarity(tf[r], tf[r + 2048], dim=0)):+.3f}')
# exact zero rows?
zr = (tf.abs().sum(1) == 0).nonzero().flatten().tolist()
print('exact-zero rows:', zr)
print('\nwhole-checkpoint scan: per tensor, count of exact zeros, of tiny values (0 < |x| < 1e-30), and the dominant exponent of tiny values')
for k in sorted(f.keys()):
    x = f.get_tensor(k)
    if x.dtype != torch.bfloat16 and x.dtype != torch.float16 and x.dtype != torch.float32:
        continue
    bits = x.view(torch.int16) if x.dtype == torch.bfloat16 else None
    xf = x.float()
    nz = int((xf == 0).sum())
    tiny = (xf != 0) & (xf.abs() < 1e-30)
    nt = int(tiny.sum())
    if nt:
        e = torch.unique(xf[tiny].abs().log2().floor()).tolist()
        print(f'  {k}: zeros {nz}, tiny {nt}, exponents {e[:8]}, total {x.numel()}')
