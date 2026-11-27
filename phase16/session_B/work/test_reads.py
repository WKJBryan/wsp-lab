import sys, time, torch
torch.set_num_threads(4)
sys.path.insert(0, '/home/user/wsp14/work')
import reads as R
t0 = time.time()
rd = R.Reader(device='cpu', capture_qk=True)
print('loaded', time.time() - t0)
tok = rd.tok
p = R.K.prompt_text(R.K.PRACTICE[1]['q'], tok)
g = "\nI need to find the integer closest to $\\sqrt[3]{6^3+8^3}$.\n6^3 = 216, 8^3 = 512, so the sum is 728.\nThe cube root of 728 is close to 9, since 9^3 = 729.\nSo the answer is 9.\n</think>\n\nThe integer closest is $\\boxed{9}$.<|im_end|>"
pids, gids = R.encode_gen(tok, p, g)
ids = torch.tensor([pids + gids])
print(len(pids), len(gids), repr(tok.decode(gids)))
t0 = time.time()
h, cap = rd.run(ids)
print('forward', time.time() - t0, h.shape, cap['res'][-1].shape, cap['attn'][3].shape, cap['gdn'][0].shape, cap['mlp'][0].shape)
gs = len(pids)
# logits vs DLA sum
sel = torch.tensor([gs - 1, gs + 5, gs + len(gids) - 2])
lg = rd.logits(h, sel)
pos = sel
toks = [R.THINK_END, R.IM_END, 100]
d = rd.dla(cap, pos, toks)
tot = d['emb'] + d['attn'].sum((0, 2)) + d['gdn'].sum((0, 2)) + d['mlp'].sum(0)
print('logits', lg[:, toks])
print('dla sum', tot)
prof = rd.profile(h, gids, gs, chunk=64)
print('act', prof['act'][:8], 'think', prof['think'][:12])
print('p_act', prof['p_act'])
print('top', [(tok.decode([int(i)]), round(float(v), 2)) for i, v in zip(prof['top'][6], prof['topl'][6])])
pr, pi, pv = rd.attn_rows(3, torch.tensor([gs, gs + 3]))
print('attn', pr.shape, pi[0, 0], pv[0, 0])
# ablation: everything zero -> logits from the embedding alone
abl = {}
for i in range(24):
    kind = 'attn' if i in R.K.FULL_LAYERS else 'gdn'
    abl[(kind, i)] = dict(pos=[], heads=list(range(8 if kind == 'attn' else 16)))
    abl[('mlp', i)] = dict(pos=[])
h2, _ = rd.run(ids, capture=False, abl=abl)
lg2 = rd.logits(h2, sel)
emb_only = rd.E.float()[toks] @ (rd.nw * cap['emb'][0][pos].float()).T / cap['emb'][0][pos].float().pow(2).mean(-1, keepdim=True).sqrt().T
print('all-ablated logits', lg2[:, toks])
print('embedding-only readout', emb_only.T)
# ablation at positions only for head 1 of layer 3
abl = {('attn', 3): dict(pos=[gs], heads=[1])}
h3, _ = rd.run(ids, capture=False, abl=abl)
print('one head one pos, delta of logits', (rd.logits(h3, sel) - lg).abs().max(1).values)
