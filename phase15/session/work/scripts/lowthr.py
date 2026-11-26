"""CPU: for the deployed detector direction (layer 14), where would a LOWER soft threshold start to
fire at flip level in fresh text?  Lists the early-region positions that reach flip level for
several thresholds, with context, so they can be judged legit or not.

Usage: python lowthr.py --theta 330,360,390,425
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--theta', default='330,360,390,425')
ap.add_argument('--s', type=float, default=20)
ap.add_argument('--c', type=float, default=5.5)
ap.add_argument('--flip', type=float, default=3.5)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(14)
keys = sorted(X['data'], key=lambda s: int(s[1:]))
G = lib.gens(1)
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
gids = {k: lib.ids_of(tok, G[k]['text']) for k in keys}


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


acts = {}
for k in keys:
    D = X['data'][k]
    x = D['x'][li].float(); T = x.shape[0]
    acts[k] = (x, regions_of(D, T))
GD = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
for th in [float(t) for t in a.theta.split(',')]:
    P = dict(d=GD['d'], w=GD['w'], theta=th, s=a.s)
    tot = {0: 0, 1: 0, 2: 0}; n = {0: 0, 1: 0, 2: 0}; traces = {0: {}, 1: {}, 2: {}}
    hits = []
    for k in keys:
        x, reg = acts[k]
        act = gl.activation(x, P)
        for r in range(3):
            sel = reg == r
            if not sel.any():
                continue
            n[r] += int(sel.sum())
            c = int(((a.c * act) >= a.flip)[sel].sum())
            tot[r] += c
            if c:
                traces[r][k] = c
                if r == 0:
                    D = X['data'][k]
                    for i in torch.nonzero(sel & ((a.c * act) >= a.flip)).flatten()[::max(1, c // 2)].tolist()[:2]:
                        t = i - D['P0']
                        hits.append((k, t, float(a.c * act[i]), dec(gids[k][max(0, t - 50):t + 1]), dec(gids[k][t + 1:t + 10])))
    print(f'\n=== soft threshold {th:.0f}: flip-level positions  early {tot[0]}/{n[0]} ({100*tot[0]/n[0]:.3f}%) in {len(traces[0])} traces | pre-loop {tot[1]} in {len(traces[1])} traces | loop {tot[2]} ({100*tot[2]/n[2]:.1f}%) in {len(traces[2])} traces')
    print('   early traces: ' + ' '.join(f'{k}:{c}' for k, c in sorted(traces[0].items(), key=lambda kv: -kv[1])))
    for k, t, w, ctx, nxt in hits[:14]:
        print(f'     {k} tok {t} write {w:.1f}: ...{ctx[-110:]!r} => {nxt!r}')
