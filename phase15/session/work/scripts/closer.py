"""For every pass-N trace: which switch closed the thinking first?
Scores of the repetition direction (d_ls) and the enumeration direction (d_enum) at the position
just before the first </think>.  Outcome by closer.

Usage: python closer.py <pass_no>
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
from transformers import AutoTokenizer
pass_no = int(sys.argv[1])
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(pass_no)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
G1 = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
d_ls = G1['d'].cuda(); w = G1['w'].cuda()
d_en = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum'].cuda()
norm = m.model.language_model.layers[14].post_attention_layernorm
P1 = dict(d=d_ls.cpu(), w=G1['w'], theta=425, s=20); P4 = dict(d=d_en.cpu(), w=G1['w'], theta=5, s=10)
rows = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    p0 = lib.prompt_ids(tok, prac[pid]['q'])
    gid = lib.ids_of(tok, g['text'])
    if THINK not in gid:
        rows.append((pid, 'none', 0, g['n_tokens'], g['ok_boxed'], g['finish'], float('nan'), float('nan'), g['text'].count('</think>')))
        continue
    k = gid.index(THINK)                 # position (in generated tokens) of the first </think>
    ids = p0 + gid[:k]
    rec = []
    h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[-1]))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    x = rec[0]
    s1, s4 = float(x @ d_ls), float(x @ d_en)
    a1 = float(gl.activation(x[None].cpu(), P1)[0]) * 5.5; a4 = float(gl.activation(x[None].cpu(), P4)[0]) * 2.0
    closer = 'N1' if a1 >= a4 else 'N4'
    rows.append((pid, closer, k, g['n_tokens'], g['ok_boxed'], g['finish'], s1, s4, g['text'].count('</think>')))
print('closer | n | ended | correct | capped | median closure token | median closure token of the correct ones')
for c in ('N1', 'N4', 'none'):
    r = [x for x in rows if x[1] == c]
    if not r:
        continue
    med = sorted(x[2] for x in r)[len(r) // 2]
    medc = sorted(x[2] for x in r if x[4])[len([x for x in r if x[4]]) // 2] if any(x[4] for x in r) else None
    print(f'  {c:4s} | {len(r):3d} | {sum(1 for x in r if x[5] == "stop"):3d} | {sum(1 for x in r if x[4]):3d} | {sum(1 for x in r if x[5] == "length"):3d} | {med:6d} | {medc}')
print('\ntraces closed by N4 (or not closed):')
for r in rows:
    if r[1] != 'N1':
        print(f'  {r[0]:5s} L{prac[r[0]]["level"][-1]} closer {r[1]:4s} closed at gen token {r[2]:5d} (total {r[3]:5d}) ok={r[4]} {r[5]:6s} d_ls {r[6]:.0f} d_enum {r[7]:+.0f} #</think> {r[8]}')
import collections
b = collections.Counter((min(r[2] // 4000, 7), r[1]) for r in rows if r[1] != 'none')
print('\nclosure token // 4000 (0 = <4k, 7 = 28k+) x closer:')
for kk in sorted(b):
    print('  ', kk, b[kk])
