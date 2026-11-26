"""Calibrate a residual write: how big must a write aligned with the output direction of </think>
be, at layer L, to raise the </think> logit at in-loop positions? (Runtime hooks only.)

For held-out traces: positions inside the loop where </think> could naturally be emitted: the token
right after a newline, and the last token of a line (next token is a newline). Per position: natural
top token and logits, then for each c: add c * v to the residual at the OUTPUT of layer L at the
last position only (v = unit vector along (1 + final norm weight) * E[</think>]).
Usage: python writecal.py --layers 13,22 --c 1,2,4,8,16,32,64 [--n 16]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--layers', default='13,22')
ap.add_argument('--c', default='1,2,4,8,16,32,64')
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--tok', default='</think>')
a = ap.parse_args()
m, tok = lib.load()
G = lib.gens(1)
prac = {p['id']: p for p in lib.practice()}
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
T = tok.convert_tokens_to_ids(a.tok)
H = m.model.language_model
s = lib.final_scale(m).cpu()
v = s * H.embed_tokens.weight[T].float().cpu()          # direction whose dot product with the final residual gives the logit
v = v / v.norm()
print(f'target token {dec([T])!r}; |scale*E| = {float((s * H.embed_tokens.weight[T].float().cpu()).norm()):.2f}')
cs = [float(x) for x in a.c.split(',')]
layers = [int(x) for x in a.layers.split(',')]
ctxs = []
for k in sorted(G, key=lambda q: int(q[1:])):
    if int(k[1:]) % 2:
        continue
    ls = loops[k]
    if not ls['loop_start'] or ls['loop_start'] > 9000:
        continue
    p_ids = lib.prompt_ids(tok, prac[k]['q'])
    ids = p_ids + lib.ids_of(tok, G[k]['text'])
    s0, P = ls['loop_start'], ls['period']
    # candidate positions in the loop: token after a newline, or last token before a newline
    cands = []
    for j in range(len(p_ids) + s0 + 2, min(len(ids) - 2, len(p_ids) + s0 + 2500)):
        prev, nxt = dec(ids[j - 1:j]), dec(ids[j + 1:j + 2])
        if '\n' in prev or '\n' in nxt:
            cands.append(j)
    if not cands:
        continue
    for j in cands[len(cands) // 3:][:2]:
        ctxs.append((k, j, ids[:j + 1], ids[j + 1]))
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} in-loop positions')
rows = {L: [] for L in layers}
for n, (k, j, ids, nxt) in enumerate(ctxs):
    kind = 'after-newline' if '\n' in dec(ids[j - 1:j]) else 'line-end'
    r = lib.run(m, ids)
    L0 = r['logits'][0]
    top = torch.topk(L0, 2)
    line = f'{k} tok {j}: {kind:12s} natural top {dec([int(top.indices[0])])!r}:{float(top.values[0]):.1f} / {dec([int(top.indices[1])])!r}:{float(top.values[1]):.1f}  target {float(L0[T]):.1f}  |'
    for L in layers:
        res = []
        for c in cs:
            def make(c=c, L=L):
                def post(mod, args, out):
                    o = out.clone()
                    o[0, -1] += (c * v).to(o.dtype).to(o.device)
                    return o
                return H.layers[L].register_forward_hook(post)
            r2 = lib.run(m, ids, hooks=[make])
            res.append(float(r2['logits'][0, T] - L0[T]))
        rows[L].append(res)
        line += f'  L{L}: ' + ' '.join(f'{x:+.1f}' for x in res) + ' |'
    print(line, flush=True)
print('\nmean gain in target logit per unit c (last c):')
for L in layers:
    R = torch.tensor(rows[L])
    print(f'  layer {L}: mean dLogit at each c: ' + ' '.join(f'{float(x):+.1f}' for x in R.mean(0)) + f'   (per unit c at c={cs[0]}: {float(R[:, 0].mean() / cs[0]):.2f})')
