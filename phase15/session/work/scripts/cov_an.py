"""CPU: would alternative detector directions catch loops that the line-start detector misses?

All in x-space (MLP input of layer L), all traces. For each candidate direction:
  loop coverage = number of looping traces whose max score within the first 800 loop tokens
                  reaches the flip level; fresh cost = early positions (and traces) at or above it.
The flip level of the deployed graft is ~448 for d_ls (soft threshold 425, width 20, write >= 3.5).
For the other directions the scale differs, so they are compared at the score that keeps the
early region clean (max over early positions of all traces except the known quasi-loops).

Usage: python cov_an.py [--layer 14]
"""
import sys, torch, argparse
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--flip', type=float, default=448)
a = ap.parse_args()
L = a.layer
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(L)
keys = sorted(X['data'], key=lambda s: int(s[1:]))
G = torch.load(f'/home/user/wsp14/work/out/graft_dirs_L{L}.pt')
QUASI = {'p50', 'p16', 'p8', 'p52'}          # genuinely repeating before an exact cycle exists


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


xs, regs, loopmax = {}, {}, {}
for k in keys:
    D = X['data'][k]
    x = D['x'][li].float(); T = x.shape[0]
    xs[k], regs[k] = x, regions_of(D, T)
    loopmax[k] = D['loop_start']
looping = [k for k in keys if loopmax[k]]
G1 = X['data'][keys[0]]['ids']
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/home/user/wsp14/sandbox/model')
dec = lambda t: tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
loop_all = torch.cat([xs[k][regs[k] == 2][::4] for k in looping])
loop_ls = torch.cat([xs[k][(regs[k] == 2) & X['data'][k]['ls'][:xs[k].shape[0]]] for k in looping])
fresh_ls = torch.cat([xs[k][(regs[k] == 0) & X['data'][k]['ls'][:xs[k].shape[0]]] for k in keys])
fresh_all = torch.cat([xs[k][regs[k] == 0][::9] for k in keys])
loop_wait = []
for k in looping:
    D = X['data'][k]
    ids = D['ids']
    T = xs[k].shape[0]
    sel = (regs[k] == 2).clone()
    idx = torch.nonzero(sel).flatten()
    nxt = torch.tensor([ids[i + 1] if i + 1 < len(ids) else -1 for i in idx.tolist()])
    sel[idx[nxt == WAIT]] = True
    sel[idx[nxt != WAIT]] = False
    loop_wait.append(xs[k][sel])
cands = {
    'd_ls (deployed)': G['d'],
    'd_all: all loop positions vs fresh': loop_all.mean(0) - fresh_all.mean(0),
    'd_wait: loop positions predicting " Wait" vs fresh': torch.cat(loop_wait).mean(0) - fresh_all.mean(0),
    'd_ls+all': None,
}
cands['d_ls+all'] = G['d'] / G['d'].norm() + cands['d_all: all loop positions vs fresh'] / cands['d_all: all loop positions vs fresh'].norm()
print(f'layer {L}; {len(looping)} looping traces; flip level for the deployed direction {a.flip:.0f}')
missed = ['p6', 'p12', 'p15', 'p18', 'p26', 'p34', 'p37', 'p47', 'p57', 'p71', 'p72', 'p75', 'p83', 'p93', 'p97', 'p121', 'p122']
for name, d in cands.items():
    lm, em, pm = {}, {}, {}
    for k in keys:
        s = xs[k] @ d
        r = regs[k]
        if loopmax[k]:
            lm[k] = float(s[r == 2].max())
        em[k] = float(s[r == 0].max()) if (r == 0).any() else float('nan')
        pm[k] = float(s[r == 1].max()) if (r == 1).any() else float('nan')
    clean_early = sorted(((v, k) for k, v in em.items() if k not in QUASI and v == v), reverse=True)[:4]
    quasi = sorted(((em[k], k) for k in QUASI), reverse=True)
    thr = max(v for v, k in clean_early) if name != 'd_ls (deployed)' else a.flip
    cov = sum(1 for k in looping if lm[k] >= thr)
    pre_hit = [(k, round(pm[k])) for k in looping if pm[k] == pm[k] and pm[k] >= thr]
    print(f'\n{name}')
    print(f'   early max of the 4 highest non-quasi-loop traces: ' + ' '.join(f'{k}:{v:.0f}' for v, k in clean_early) + f'  -> clean threshold {thr:.0f}')
    print(f'   quasi-loop traces early max: ' + ' '.join(f'{k}:{v:.0f}' for v, k in quasi))
    print(f'   loops caught within first 800 loop tokens at that threshold: {cov}/{len(looping)}; pre-loop hits: {pre_hit}')
    print('   loop maxima of loops the deployed graft misses: ' + ' '.join(f'{k}:{lm[k]:.0f}' for k in missed))
