"""CPU: with soft threshold 390 (instead of 425), which PRE-LOOP positions of pass-1 traces reach
flip level, what was the model saying there, and what did the same problems give in pass 2?

Usage: python thr390.py
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(14)
keys = sorted(X['data'], key=lambda s: int(s[1:]))
G1 = lib.gens(1); G2 = lib.gens(2)
prac = {p['id']: p for p in lib.practice()}
GD = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
gids = {k: lib.ids_of(tok, G1[k]['text']) for k in keys}


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


for th in (390, 360):
    P = dict(d=GD['d'], w=GD['w'], theta=th, s=20)
    print(f'\n===== soft threshold {th}: first flip-level position in the PRE-LOOP region (and EARLY region), per trace')
    for k in keys:
        D = X['data'][k]
        x = D['x'][li].float(); T = x.shape[0]
        reg = regions_of(D, T)
        act = gl.activation(x, P)
        hit = (5.5 * act >= 3.5) & (reg != 2)
        idx = torch.nonzero(hit).flatten()
        if not idx.numel():
            continue
        i = int(idx[0])
        t = i - D['P0']
        g2 = G2[k]
        ls = D['loop_start']
        if ls:
            later = int(torch.nonzero((5.5 * act >= 3.5) & (reg == 2)).flatten()[0]) - D['P0'] - ls if ((5.5 * act >= 3.5) & (reg == 2)).any() else None
            where = f'loop starts at {ls}, i.e. {ls - t} tokens later; in-loop flip-level {later} loop-tokens after loop start'
        else:
            where = 'no exact loop in pass 1'
        print(f'  {k} [{prac[k]["level"][-1]}] gen tok {t} ({where}): '
              f'pass-2 result {g2["ok_boxed"]} pred {str(g2["pred_boxed"])[:30]!r} gold {prac[k]["gold"][:25]!r}')
        print(f'       ...{dec(gids[k][max(0, t - 200):t + 1])!r}'[-330:])
