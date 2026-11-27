import sys, torch
sys.path.insert(0, '/home/user/wsp14/work')
import reads as R
rd = R.Reader('model', device='cuda')
tok = rd.tok
p = R.K.prompt_text(R.K.PRACTICE[1]['q'], tok)
g = "\nI need to find the integer closest to $\\sqrt[3]{6^3+8^3}$.\n6^3 = 216, 8^3 = 512, so the sum is 728.\nThe cube root of 728 is close to 9, since 9^3 = 729.\nSo the answer is 9.\n</think>\n\nThe integer closest is $\\boxed{9}$.<|im_end|>"
pids, gids = R.encode_gen(tok, p, g)
ids = torch.tensor([pids + gids])
h, cap = rd.run(ids)
res = cap['res'][-1].float()
emb = cap['emb'].float()
tot = emb.clone()
for i, layer in enumerate(rd.tm.layers):
    if i in R.K.FULL_LAYERS:
        tot = tot + layer.self_attn.o_proj(cap['attn'][i]).float()[0]
    else:
        tot = tot + layer.linear_attn.out_proj(cap['gdn'][i]).float()[0]
    tot = tot + cap['mlp'][i].float()
print('sum components vs res: max abs diff', (tot - res).abs().max().item(), 'res norm', res.norm().item())
rms = res.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
hn = rd.nw[None] * res / rms
print('h vs nw*res/rms: max abs diff', (h.float() - hn).abs().max().item(), 'h norm', h.float().norm().item())
print('ratio', (h.float() / hn).mean().item())
# norm weights
print('norm weight stats', rd.nw.mean().item(), rd.nw.min().item(), rd.nw.max().item())
lg = rd.logits(h, torch.tensor([5, 10], device='cuda'))
dl = rd.dla(cap, [R.THINK_END, 100, 101])
print('true logits', lg[:, [R.THINK_END, 100, 101]])
print('dla total', dl['total'][[5, 10]])
