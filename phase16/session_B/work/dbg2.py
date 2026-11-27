import sys, torch
sys.path.insert(0, '/home/user/wsp14/work')
import reads as R
rd = R.Reader('model', device='cuda')
tok = rd.tok
p = R.K.prompt_text(R.K.PRACTICE[1]['q'], tok)
g = "\nI need to find the integer closest to $\\sqrt[3]{6^3+8^3}$.\n6^3 = 216, 8^3 = 512, so the sum is 728.\nSo the answer is 9.\n</think>\n\nThe integer closest is $\\boxed{9}$.<|im_end|>"
pids, gids = R.encode_gen(tok, p, g)
ids = torch.tensor([pids + gids])
h, cap = rd.run(ids)
prev = cap['emb'].float()
for i, layer in enumerate(rd.tm.layers):
    if i in R.K.FULL_LAYERS:
        o = layer.self_attn.o_proj(cap['attn'][i]).float()[0]
    else:
        o = layer.linear_attn.out_proj(cap['gdn'][i]).float()[0]
    m = cap['mlp'][i].float()
    new = prev + o + m
    d = (new - cap['res'][i].float()).abs().max().item()
    print(i, 'layer', type(layer).__name__, 'diff', round(d, 4), 'res norm', round(cap['res'][i].float().norm().item(), 2),
          'o norm', round(o.norm().item(), 2), 'm norm', round(m.norm().item(), 2), cap['res'][i].dtype)
    prev = cap['res'][i].float()
res = cap['res'][-1].float()
rms = res.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
hn = rd.nw[None] * res / rms
dd = (h.float() - hn).abs().max(1).values
print('h vs recomputed: worst positions', torch.topk(dd, 5))
print('res[0]', res[0, :6], 'h[0]', h[0, :6].float())
