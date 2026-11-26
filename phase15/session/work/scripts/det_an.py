"""Detector operating analysis over ALL positions with three regions kept apart.

  early : before (loop_start - 2500) in looping traces; everything in traces that never looped
  pre   : the last 2500 tokens before the loop (the repetitive re-check phase)
  loop  : inside the loop
Direction d per layer is a mean difference trained on the TRAIN half of the line-start data
(odd ids) and evaluated on the held-out half (even ids = the all-position data).

Target definitions compared:
  T1  LOOP line starts            vs FRESH line starts
  T2  LOOP & duplicate-line starts vs FRESH line starts
  T3  LOOP & "Wait"-top starts    vs FRESH line starts whose first token is " Wait"

Usage: python det_an.py [--layers 12,13,14,16]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--layers', default='12,13,14,16')
ap.add_argument('--show', type=int, default=12)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
G = lib.gens(1)
prac = {p['id']: p for p in lib.practice()}
LS = torch.load('/home/user/wsp14/work/out/linestart_1.pt')
AP = torch.load('/home/user/wsp14/work/out/allpos_1.pt', weights_only=False)
keys = sorted(LS, key=lambda s: int(s[1:]))
cat = lambda f: torch.cat([f(LS[k]) for k in keys])
regions = cat(lambda d: torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in d['region']]))
first = cat(lambda d: d['first']); p_top = cat(lambda d: d['p_top']); dup = cat(lambda d: d['dup'])
trace = torch.cat([torch.full((len(LS[k]['region']),), i) for i, k in enumerate(keys)])
looping = torch.tensor([any(r == 'loop' for r in LS[k]['region']) for k in keys])[trace]
tr = torch.tensor([i % 2 == 1 for i in range(len(keys))])[trace]       # train = odd ids
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
FRESH = (regions == 0) | ((regions == 1) & ~looping)
held = sorted(AP['data'], key=lambda s: int(s[1:]))                      # even ids
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


names = ['early', 'pre', 'loop']
for L in [int(x) for x in a.layers.split(',')]:
    li = AP['layers'].index(L)
    X = torch.cat([LS[k]['resid'][L] for k in keys])
    LOOP = tr & (regions == 2); FR = tr & FRESH
    dirs = {
        'T1 loop vs fresh': X[LOOP].mean(0) - X[FR].mean(0),
        'T2 loop&dup vs fresh': X[LOOP & (dup == 1)].mean(0) - X[FR].mean(0),
        'T3 loop&Wait-top vs fresh Wait': X[LOOP & (first == WAIT) & (p_top > 0.9)].mean(0) - X[FR & (first == WAIT)].mean(0),
    }
    for name, d in dirs.items():
        s_tr = X @ d
        thr = [float(torch.quantile(s_tr[regions == 2], q)) for q in (0.25, 0.50, 0.75)]
        print(f'\n=== layer {L} | {name} | loop line-start score q25/q50/q75 = ' + ' '.join(f'{t:+.2f}' for t in thr) +
              f' | held-out-loop line starts above q50: {float(((s_tr[~tr & (regions == 2)]) > thr[1]).float().mean()):.2f}')
        cnt = {r: torch.zeros(3, dtype=torch.long) for r in range(3)}      # per region: [n, n>q25, n>q50]
        tot = {r: torch.zeros(3, dtype=torch.long) for r in range(3)}
        per_trace_early = []
        hits = []
        for k in held:
            D = AP['data'][k]
            R = D['resid'][li].float(); T = R.shape[0]
            s = R @ d
            reg = regions_of(D, T)
            for r in range(3):
                sel = reg == r
                tot[r][0] += int(sel.sum()); tot[r][1] += int((s[sel] > thr[0]).sum()); tot[r][2] += int((s[sel] > thr[1]).sum())
            e = reg == 0
            per_trace_early.append((float(s[e].max()) if e.any() else float('nan'), k))
            for i in torch.nonzero(e & (s > thr[1])).flatten().tolist():
                hits.append((float(s[i]), k, i))
        for r in range(3):
            n, n25, n50 = (int(v) for v in tot[r])
            print(f'     {names[r]:5s}: positions {n:7d}  above q25: {n25:6d} ({100*n25/max(n,1):.2f}%)  above q50: {n50:5d} ({100*n50/max(n,1):.3f}%)')
        top = sorted(per_trace_early, reverse=True)[:10]
        print('     highest early maxima (trace:score): ' + ' '.join(f'{k}:{v:+.2f}' for v, k in top))
        if name.startswith('T1') and L == int(a.layers.split(',')[0]):
            for v, k, i in sorted(hits, reverse=True)[:a.show]:
                D = AP['data'][k]
                print(f'       early hit {v:+.2f} {k} tok {i - D["P0"]}: ...{dec(lib.ids_of(tok, G[k]["text"])[max(0, i - D["P0"] - 22):i - D["P0"] + 1])!r} -> {dec(lib.ids_of(tok, G[k]["text"])[i - D["P0"] + 1:i - D["P0"] + 8])!r}')
