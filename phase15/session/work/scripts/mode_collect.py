"""Collect the layer-L MLP input x on pass-2 traces, labelled by phase:
  0 = thinking (before the first </think>), 1 = final answer (after it).
Also flags positions whose NEXT token is a "Wait" variant, in both phases.

Usage: python mode_collect.py --pass 2 --layer 14
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--pass', dest='pass_no', type=int, default=2)
ap.add_argument('--layer', type=int, default=14)
ap.add_argument('--every_think', type=int, default=32)
ap.add_argument('--every_final', type=int, default=8)
ap.add_argument('--save', default='')
a = ap.parse_args()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(a.pass_no)
THINK = tok.convert_tokens_to_ids('</think>')
WAITV = {tok(t, add_special_tokens=False)['input_ids'][0] for t in [' Wait', 'Wait', ' wait', 'wait']}
print('wait-variant ids:', WAITV)
norm = m.model.language_model.layers[a.layer].post_attention_layernorm
data = {}
for n, pid in enumerate(sorted(G, key=lambda s: int(s[1:]))):
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    if THINK not in ids:
        continue
    c0 = ids.index(THINK)                      # position of the first </think> token; later positions are final phase
    T = len(ids)
    is_wait = torch.zeros(T, dtype=torch.bool)
    for t in range(T - 1):
        if ids[t + 1] in WAITV:
            is_wait[t] = True
    sel = torch.zeros(T, dtype=torch.bool)
    sel[:c0:a.every_think] = True
    sel[c0::a.every_final] = True
    sel |= is_wait
    pos = torch.nonzero(sel).flatten().cuda()
    rec = []
    h = norm.register_forward_hook(lambda mod, args, out: rec.append(out[0].float()[pos].half().cpu()))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    p = pos.cpu()
    data[pid] = dict(x=rec[0], phase=(p >= c0).to(torch.uint8), is_wait=is_wait[p], ok=g['ok_boxed'], finish=g['finish'], ntok=g['n_tokens'])
    if n % 16 == 0:
        print(f'  {n}/128 traces; last T={T} closure at {c0}; samples {len(p)} (final-phase wait positions {int((is_wait[p] & (p >= c0)).sum())})', flush=True)
out = a.save or f'/home/user/wsp14/work/out/mode_{a.pass_no}_L{a.layer}.pt'
torch.save(dict(layer=a.layer, data=data), out)
tot = sum((d['phase'] == 1).sum().item() for d in data.values())
print(f'saved {out}: {len(data)} traces; final-phase samples {tot}; final-phase wait-next samples {sum(int((d["is_wait"] & (d["phase"] == 1)).sum()) for d in data.values())}')
