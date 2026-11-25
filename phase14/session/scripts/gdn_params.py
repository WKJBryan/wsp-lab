import torch
import torch.nn.functional as F
from safetensors import safe_open
f = safe_open('/home/user/wsp14/sandbox/model/model.safetensors-00001-of-00001.safetensors', framework='pt')
def lst(t, fmt='%.2f'):
    return '[' + ','.join(fmt % v for v in t.tolist()) + ']'
for L in range(24):
    if L in (3,7,11,15,19,23): continue
    p = 'model.language_model.layers.%d.linear_attn.' % L
    A_log = f.get_tensor(p+'A_log').float()
    dtb = f.get_tensor(p+'dt_bias').float()
    a_w = f.get_tensor(p+'in_proj_a.weight').float()
    a_rms = a_w.pow(2).mean(dim=1).sqrt()
    g = -torch.exp(A_log) * F.softplus(dtb)
    decay = torch.exp(g)
    print('L%2d A_log %s' % (L, lst(A_log)))
    print('    dt_bias %s' % lst(dtb))
    print('    per-token decay (a=x): %s' % lst(decay, '%.3f'))
    print('    in_proj_a row-rms: %s' % lst(a_rms, '%.3f'))
