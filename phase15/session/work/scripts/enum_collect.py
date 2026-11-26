"""Collect layer-L MLP inputs for a runaway-ENUMERATION detector.

Positives: every 4th position of the last 6000 tokens of traces that ended in an enumeration
(runaway list with an incrementing number, never an exact cycle): pass 2 for
p28 p29 p31 p37 p44 p57 p99 p123 and pass 1 for p16 p64 p85 p106 p112.
Negatives: every 16th position of the first 3000 generated tokens of all pass-1 traces (fresh).

Usage: python enum_collect.py --layer 14
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--layer', type=int, default=14)
a = ap.parse_args()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
prac = {p['id']: p for p in lib.practice()}
POS = {2: ['p28', 'p29', 'p31', 'p37', 'p44', 'p57', 'p99', 'p123'], 1: ['p16', 'p64', 'p85', 'p106', 'p112']}
norm = m.model.language_model.layers[a.layer].post_attention_layernorm
out = {}
for pass_no, pids in POS.items():
    G = lib.gens(pass_no)
    for n, pid in enumerate(pids):
        g = G[pid]
        ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
        T = len(ids)
        lo = max(0, T - 6000)
        pos = torch.arange(lo + 3, T - 2, 4).cuda()
        rec = []
        h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[pos].half().cpu()))
        try:
            m.model(input_ids=torch.tensor([ids], device='cuda'))
        finally:
            h.remove()
        out[f'{pass_no}:{pid}'] = dict(x=rec[0], t=pos.cpu() - len(ids), kind='enum', finish=g['finish'], ntok=g['n_tokens'])
        print(f'  pass {pass_no} {pid}: T={T}, finish={g["finish"]}, {len(pos)} samples; tail {tok.decode(ids[-60:], skip_special_tokens=False)!r}'[:230], flush=True)
G = lib.gens(1)
neg = {}
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    ids = ids[:len(ids) if g['finish'] == 'stop' and g['n_tokens'] < 3000 else min(len(ids), 60 + 3000)]
    pos = torch.arange(len(lib.prompt_ids(tok, prac[pid]['q'])) + 30, len(ids) - 2, 16).cuda()
    if pos.numel() < 5:
        continue
    rec = []
    h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[pos].half().cpu()))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    neg[pid] = rec[0]
torch.save(dict(layer=a.layer, pos=out, neg=neg), f'/home/user/wsp14/work/out/enum_L{a.layer}.pt')
print(f'saved: positives {sum(v["x"].shape[0] for v in out.values())} samples from {len(out)} traces; negatives {sum(v.shape[0] for v in neg.values())} from {len(neg)} traces')
