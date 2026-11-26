"""Where inside a loop does the detector fire? For a few traces show the loop positions above the
threshold with their context, and summarise what kind of position they are.

Usage: python det_loop.py <layer> <threshold> [n_traces 5] [per_trace 6]
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
L = int(sys.argv[1]); thr = float(sys.argv[2])
n_traces = int(sys.argv[3]) if len(sys.argv) > 3 else 5
per_trace = int(sys.argv[4]) if len(sys.argv) > 4 else 6
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
G = lib.gens(1)
LS = torch.load('/home/user/wsp14/work/out/linestart_1.pt')
AP = torch.load('/home/user/wsp14/work/out/allpos_1.pt', weights_only=False)
keys = sorted(LS, key=lambda s: int(s[1:]))
cat = lambda f: torch.cat([f(LS[k]) for k in keys])
regions = cat(lambda d: torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in d['region']]))
trace = torch.cat([torch.full((len(LS[k]['region']),), i) for i, k in enumerate(keys)])
looping = torch.tensor([any(r == 'loop' for r in LS[k]['region']) for k in keys])[trace]
tr = torch.tensor([i % 2 == 1 for i in range(len(keys))])[trace]
FRESH = (regions == 0) | ((regions == 1) & ~looping)
X = torch.cat([LS[k]['resid'][L] for k in keys])
d = X[tr & (regions == 2)].mean(0) - X[tr & FRESH].mean(0)
held = sorted(AP['data'], key=lambda s: int(s[1:]))
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
li = AP['layers'].index(L)
kinds = {}
shown = 0
for k in held:
    D = AP['data'][k]
    if not D['loop_start']:
        continue
    R = D['resid'][li].float(); T = R.shape[0]
    s = R @ d
    s0 = D['P0'] + D['loop_start']
    lo = torch.zeros(T, dtype=torch.bool); lo[s0:] = True
    idx = torch.nonzero(lo & (s > thr)).flatten()
    gids = lib.ids_of(tok, G[k]['text'])
    for i in idx.tolist():
        t = i - D['P0']
        prev = dec(gids[t - 1:t]) if t >= 1 else ''
        nxt = dec(gids[t + 1:t + 2]) if t + 1 < len(gids) else ''
        kind = ('after-newline' if '\n' in prev else 'line-end(next is newline)' if '\n' in nxt else 'mid-line')
        kinds[kind] = kinds.get(kind, 0) + 1
    if shown < n_traces and len(idx):
        shown += 1
        print(f'\n--- {k}: loop at {D["loop_start"]}, {int(lo.sum())} loop tokens, {len(idx)} above {thr:+.2f} (first at loop token {int(idx[0]) - s0})')
        step = max(1, len(idx) // per_trace)
        for i in idx[::step][:per_trace].tolist():
            t = i - D['P0']
            print(f'   {float(s[i]):+.2f} loop tok {t - D["loop_start"]:5d}: ...{dec(gids[max(0, t - 45):t + 1])!r} => {dec(gids[t + 1:t + 12])!r}')
tot = sum(kinds.values())
print('\nkinds of firing positions in loops (all held-out looping traces):', {k: f'{v} ({100*v/tot:.0f}%)' for k, v in kinds.items()})
