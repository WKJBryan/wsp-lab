import sys, torch
sys.path.insert(0, '/home/user/wsp14/work')
import reads as R
rd = R.Reader('model', device='cuda')
tok = rd.tok
store = {}
h1 = rd.tm.norm.register_forward_pre_hook(lambda m, args: store.__setitem__('norm_in', args[0].detach()[0].clone()))
h2 = rd.tm.layers[0].register_forward_pre_hook(lambda m, args: store.__setitem__('l0_in', args[0].detach()[0].clone()))
h3 = rd.tm.layers[23].mlp.register_forward_hook(lambda m, a, o: store.__setitem__('l23_mlp_out', o.detach()[0].clone()))
p = R.K.prompt_text(R.K.PRACTICE[1]['q'], tok)
g = "\nI need to find the integer closest to $\\sqrt[3]{6^3+8^3}$.\n6^3 = 216, 8^3 = 512, so the sum is 728.\nSo the answer is 9.\n</think>\n\nThe integer closest is $\\boxed{9}$.<|im_end|>"
pids, gids = R.encode_gen(tok, p, g)
ids = torch.tensor([pids + gids])
h, cap = rd.run(ids)
print('norm input vs res[-1]', (store['norm_in'].float() - cap['res'][-1].float()).abs().max().item())
print('layer0 input vs emb', (store['l0_in'].float() - cap['emb'].float()).abs().max().item(),
      'ratio', (store['l0_in'].float().norm() / cap['emb'].float().norm()).item())
print('l23 mlp out vs captured', (store['l23_mlp_out'].float() - cap['mlp'][23].float()).abs().max().item())
e = store['l0_in'].float(); b = cap['emb'].float()
print('l0_in[0,:6]', e[0, :6]); print('emb[0,:6]', b[0, :6])
print('per-dim ratio first tokens', (e[1, :8] / b[1, :8]))
print('l0_in dtype', store['l0_in'].dtype)
res = cap['res'][-1].float()
rms = res.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
hn = rd.nw[None] * res / rms
print('h[0,:6]', h[0, :6].float()); print('hn[0,:6]', hn[0, :6]); print('nw[:6]', rd.nw[:6])
print('h dtype', h.dtype)
