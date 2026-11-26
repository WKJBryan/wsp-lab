"""Runaway-enumeration detector: (1) train the direction on all 13 runaway traces + even-id fresh
samples (closed-form ridge LDA); (2) score profile along each pass-2 runaway trace, from 24k
tokens before the end, to see how the score builds up; (3) score over ALL positions of all pass-1
traces (xpos_1) by region, to find the highest values reached in ordinary reasoning.

Usage: python enum_profile.py
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
E = torch.load('/home/user/wsp14/work/out/enum_L14.pt', weights_only=False)
pos, neg = E['pos'], E['neg']
nkeys = sorted(neg, key=lambda s: int(s[1:]))
P = torch.cat([v['x'] for v in pos.values()]).float()
Ntr = torch.cat([neg[k] for k in nkeys if int(k[1:]) % 2 == 0]).float()
Nte = torch.cat([neg[k] for k in nkeys if int(k[1:]) % 2 == 1]).float()
mu_p, mu_n = P.mean(0), Ntr.mean(0)
X = torch.cat([P, Ntr]); Xc = X - X.mean(0)
C = Xc.t() @ Xc / (X.shape[0] - 2)
de = torch.linalg.solve(C + 1e-3 * float(torch.trace(C)) / 1024 * torch.eye(1024), mu_p - mu_n)
# rescale so that the held-out fresh maximum is 0 and the runaway median is 100
s_te = Nte @ de; s_p = P @ de
de = de * (100.0 / (float(s_p.median()) - float(s_te.max())))
s_te = Nte @ de
print(f'direction |de| = {float(de.norm()):.4f}; fresh (train side) max {float((Ntr @ de).max()):.1f} q99 {float(torch.quantile(Ntr @ de, 0.99)):.1f} median {float((Ntr @ de).median()):.1f}; '
      f'fresh (held-out side) max {float(s_te.max()):.1f} q99 {float(torch.quantile(s_te, 0.99)):.1f} | runaway median {float((P @ de).median()):.1f} min {float((P @ de).min()):.1f}')
torch.save(dict(d_enum=de), '/home/user/wsp14/work/out/enum_dir_L14.pt')
# ---------- (2) profile along the pass-2 runaway traces
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
prac = {p['id']: p for p in lib.practice()}
norm = m.model.language_model.layers[14].post_attention_layernorm
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
G2 = lib.gens(2)
de_g = de.cuda()
print('\n== score profile along runaway traces (median score per 2000-token bin, oldest first; T = total tokens incl. prompt)')
for pid in ['p28', 'p29', 'p31', 'p37', 'p44', 'p57', 'p99', 'p123']:
    g = G2[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    T = len(ids)
    lo = max(0, T - 24000)
    pos_idx = torch.arange(lo + 3, T - 2, 8).cuda()
    rec = []
    h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[pos_idx]))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    sc = (rec[0] @ de_g).cpu()
    t = pos_idx.cpu()
    bins = [(lo + 2000 * i, lo + 2000 * (i + 1)) for i in range((T - lo) // 2000 + 1)]
    prof = ' '.join(f'{float(sc[(t >= a) & (t < b)].median()) if ((t >= a) & (t < b)).any() else float("nan"):.0f}' for a, b in bins)
    first = next((int(tt) for tt, ss in zip(t.tolist(), sc.tolist()) if ss >= 60), None)
    print(f'  {pid}: T={T} | median score per 2000-token bin: {prof}')
    print(f'        first token where score >= 60: {first} (generated token {None if first is None else first - (T - g["n_tokens"])})')
    if first:
        k = first - (T - g['n_tokens'])
        gid = lib.ids_of(tok, g['text'])
        print(f'        context there: ...{dec(gid[max(0, k - 420):k + 1])!r}'[:900])
# ---------- (3) all positions of all pass-1 traces
X1 = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X1['layers'].index(14)
keys = sorted(X1['data'], key=lambda s: int(s[1:]))


def regions_of(D, T):
    r = torch.zeros(T, dtype=torch.long)
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        r[max(0, s0 - 2500):s0] = 1
        r[s0:] = 2
    return r


print('\n== scores over ALL positions of pass-1 traces by region (threshold candidates 40 / 60 / 80 / 100)')
agg = {r: [0, 0, 0, 0, 0] for r in range(3)}
per = {r: [] for r in range(3)}
for k in keys:
    D = X1['data'][k]
    x = D['x'][li].float(); T = x.shape[0]
    s = x @ de
    reg = regions_of(D, T)
    for r in range(3):
        sel = reg == r
        if not sel.any():
            continue
        agg[r][0] += int(sel.sum())
        for j, thr in enumerate([40, 60, 80, 100]):
            agg[r][1 + j] += int((s[sel] >= thr).sum())
        per[r].append((float(s[sel].max()), k))
for r, name in enumerate(['early', 'pre-loop', 'loop']):
    n, a, b, c, d = agg[r]
    print(f'  {name:8s}: positions {n:7d}  >=40: {a:6d} ({100*a/max(n,1):.3f}%)  >=60: {b:6d} ({100*b/max(n,1):.3f}%)  >=80: {c:5d}  >=100: {d:5d}')
    print(f'           highest maxima: ' + ' '.join(f'{k}:{v:.0f}' for v, k in sorted(per[r], reverse=True)[:14]))
