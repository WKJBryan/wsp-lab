"""Show the highest-scoring EARLY-region positions for a given layer/target, one block per trace,
with context, so it can be judged whether they are genuine repetition or false alarms.

Usage: python det_show.py <layer> [T1|T2|T3] [per_trace 3] [n_traces 8]
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
L = int(sys.argv[1]); which = sys.argv[2] if len(sys.argv) > 2 else 'T1'
per_trace = int(sys.argv[3]) if len(sys.argv) > 3 else 3
n_traces = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
G = lib.gens(1)
LS = torch.load('/home/user/wsp14/work/out/linestart_1.pt')
AP = torch.load('/home/user/wsp14/work/out/allpos_1.pt', weights_only=False)
keys = sorted(LS, key=lambda s: int(s[1:]))
cat = lambda f: torch.cat([f(LS[k]) for k in keys])
regions = cat(lambda d: torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in d['region']]))
first = cat(lambda d: d['first']); p_top = cat(lambda d: d['p_top']); dup = cat(lambda d: d['dup'])
trace = torch.cat([torch.full((len(LS[k]['region']),), i) for i, k in enumerate(keys)])
looping = torch.tensor([any(r == 'loop' for r in LS[k]['region']) for k in keys])[trace]
tr = torch.tensor([i % 2 == 1 for i in range(len(keys))])[trace]
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
FRESH = (regions == 0) | ((regions == 1) & ~looping)
held = sorted(AP['data'], key=lambda s: int(s[1:]))
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
X = torch.cat([LS[k]['resid'][L] for k in keys])
if which == 'T1':
    d = X[tr & (regions == 2)].mean(0) - X[tr & FRESH].mean(0)
elif which == 'T2':
    d = X[tr & (regions == 2) & (dup == 1)].mean(0) - X[tr & FRESH].mean(0)
else:
    d = X[tr & (regions == 2) & (first == WAIT) & (p_top > 0.9)].mean(0) - X[tr & FRESH & (first == WAIT)].mean(0)
li = AP['layers'].index(L)
thr = float(torch.quantile((X @ d)[regions == 2], 0.5))
print(f'layer {L} {which}: threshold (loop line-start median) {thr:+.2f}')
res = []
for k in held:
    D = AP['data'][k]
    R = D['resid'][li].float(); T = R.shape[0]
    s = R @ d
    s0 = D['P0'] + D['loop_start'] if D['loop_start'] else T
    early_end = max(0, s0 - 2500)
    e = torch.zeros(T, dtype=torch.bool); e[:early_end] = True
    if int(e.sum()) < per_trace:
        continue
    v, i = torch.topk(s[e], per_trace)
    idx = torch.nonzero(e).flatten()
    res.append((float(v[0]), k, [(float(v[j]), int(idx[i[j]])) for j in range(per_trace)], D['loop_start']))
res.sort(reverse=True)
for top, k, hits, ls in res[:n_traces]:
    gids = lib.ids_of(tok, G[k]['text'])
    D = AP['data'][k]
    print(f'\n--- {k}: loop_start {ls}; early max {top:+.2f}')
    for v, i in hits:
        t = i - D['P0']
        print(f'   {v:+.2f} tok {t}: ...{dec(gids[max(0, t - 60):t + 1])!r}  =>  {dec(gids[t + 1:t + 10])!r}')
