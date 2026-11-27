import sys, time, torch
sys.path.insert(0, '/home/user/wsp14/work')
import edits as ED, reads as R
from val_edit import make_edit
rd = R.Reader('base_model', device='cuda')
A = ED.LiveAccess(rd)
make_edit('R1')(A)
for e in 'H15.7x-1+H19.0x-1+H15.6x-1+H23.0x-1+H19.2x-1'.split('+'):
    make_edit(e)(A)
r = R.gen_rows(2)['p0']
tok = rd.tok
pids, gids = R.encode_gen(tok, R.K.prompt_text(R.K.PRACTICE_BY_ID['p0']['q'] if hasattr(R.K, 'PRACTICE_BY_ID') else [p for p in R.K.PRACTICE if p['id'] == 'p0'][0]['q'], tok), r['text'])
ids = torch.tensor([pids + gids])
torch.cuda.synchronize(); t0 = time.time()
h, _ = rd.run(ids, capture=False)
torch.cuda.synchronize(); t1 = time.time()
prof = rd.profile(h, gids, len(pids))
torch.cuda.synchronize(); t2 = time.time()
sel = torch.arange(len(pids) - 1, len(pids) - 1 + 1500, device='cuda')
lg = rd.logits(h, sel); lp = torch.log_softmax(lg, -1); tb = lp.topk(20, -1)
torch.cuda.synchronize(); t3 = time.time()
x = float((torch.exp(tb.values) * (tb.values - torch.gather(lp, -1, tb.indices))).sum(-1).mean())
torch.cuda.synchronize(); t4 = time.time()
print('fwd', round(t1 - t0, 2), 'profile', round(t2 - t1, 2), 'logits', round(t3 - t2, 2), 'kl', round(t4 - t3, 2))
print('nan in h:', bool(torch.isnan(h).any()), 'max logit', float(lg.max()), 'min', float(lg.min()))
print('tail act mean', prof['act'][-1000:].mean().item(), 'p_act', prof['p_act'][-1000:].mean().item())
