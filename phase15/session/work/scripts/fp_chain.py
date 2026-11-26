"""Do the detectors stay active one position after a forced token?
At final-phase runaway positions (pass 2, re-closure positions): scores of the repetition
direction (d_ls) and the enumeration direction (d_enum) at the firing position and at the next
position after appending a forced token.

Usage: python fp_chain.py [--n 16]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--maxctx', type=int, default=16000)
a = ap.parse_args()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(2)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
d_ls = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')['d'].cuda()
d_en = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum'].cuda()
norm = m.model.language_model.layers[14].post_attention_layernorm
forced = {'\\\\ (59)': T('\\'), ' \\\\ (1088)': T(' \\'), 'ĊĊ (271)': T('\n\n'), '</think>': [THINK]}
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    if g['finish'] != 'length' or '</think>' not in g['text']:
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + T(g['text'])
    first = ids.index(THINK)
    later = [i for i in range(first + 20, min(len(ids) - 40, a.maxctx)) if ids[i] == THINK]
    for i in later[1:2]:
        ctxs.append((pid, i - 1, ids[:i]))
    if len(ctxs) >= a.n:
        break
res = {name: [] for name in forced}; base = []
for pid, p, ids in ctxs:
    out = {}
    for name, ftok in [('', [])] + list(forced.items()):
        full = ids + ftok
        rec = []
        h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[-1]))
        try:
            m.model(input_ids=torch.tensor([full], device='cuda'))
        finally:
            h.remove()
        out[name] = (float(rec[0] @ d_ls), float(rec[0] @ d_en))
        if name == ' \\\\ (1088)':
            # natural continuation after the forced " \\" : what the model writes next
            lg = m(input_ids=torch.tensor([full], device='cuda')).logits[0, -1].float()
            pr = torch.softmax(lg, 0)
            v, i = torch.topk(lg, 6)
            out['next'] = ' '.join(f'{tok.decode([int(t)])!r}:{float(pr[t]):.2f}' for t, x in zip(i, v))
    base.append(out[''])
    print(f'  {pid}: at firing position  d_ls {out[""][0]:.0f}  d_enum {out[""][1]:+.0f} | ' + ' | '.join(f'after {nm}: d_ls {out[nm][0]:.0f} d_enum {out[nm][1]:+.0f}' for nm in forced))
    print(f'        after forced " \\": {out["next"]}')
import statistics
print(f'\nmedian at firing position: d_ls {statistics.median(b[0] for b in base):.0f}, d_enum {statistics.median(b[1] for b in base):+.0f}')
