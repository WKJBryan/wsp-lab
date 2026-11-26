"""Emulate the grafted neuron with runtime hooks and inspect what happens at and after the flip.

(A) in-loop positions (held-out traces) where the write first reaches flip level
(B) natural closure positions of traces that closed (token right before </think>)
(C) mid-reasoning line ends of traces that closed
For (A) also continue with forced tokens: "</think>", then newlines, then "The final answer is",
and report score / write / top tokens at each step with the graft active.

Usage: python emul.py [--theta 425] [--s 20] [--c 5.5] [--n 10]
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
ap = argparse.ArgumentParser()
ap.add_argument('--theta', type=float, default=425); ap.add_argument('--s', type=float, default=20); ap.add_argument('--c', type=float, default=5.5)
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--token', default='</think>')
a = ap.parse_args()
P = gl.load_params(theta=a.theta, s=a.s, c=a.c, token=a.token)
m, tok = lib.load()
P['_m'] = m
THINK = tok.convert_tokens_to_ids('</think>')
G = lib.gens(1)
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(P['layer'])


def top(L, k=4):
    p = torch.softmax(L, 0)
    v, i = torch.topk(L, k)
    return ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}({float(p[t]):.2f})' for t, x in zip(i, v)) + f'  [</think> {float(L[THINK]):.1f} p={float(p[THINK]):.3f}]'


loop_ctx, closed_ctx, fresh_ctx = [], [], []
for k in sorted(X['data'], key=lambda s: int(s[1:])):
    if int(k[1:]) % 2:
        continue
    D = X['data'][k]
    x = D['x'][li].float(); T = x.shape[0]
    ids = D['ids'].tolist()
    if D['loop_start']:
        s0 = D['P0'] + D['loop_start']
        act = gl.activation(x[s0:], P)
        hit = torch.nonzero(a.c * act >= 3.5).flatten()
        if hit.numel() and len(ids) <= 11000 and len(loop_ctx) < a.n:
            loop_ctx.append((k, s0 + int(hit[0]), ids))
    elif D['finish'] == 'stop' and THINK in ids:
        it = ids.index(THINK)
        closed_ctx.append((k, it - 1, ids))
        # a mid-reasoning line end
        ends = [j for j in range(D['P0'] + 400, it - 200) if dec(ids[j + 1:j + 2]) == '\n']
        if ends:
            fresh_ctx.append((k, ends[len(ends) // 2], ids))
print(f'{len(loop_ctx)} loop contexts, {len(closed_ctx)} natural closures, {len(fresh_ctx)} fresh line ends')
print('\n===== (B) natural closure positions and (C) fresh line ends: graft must not change anything =====')
for name, ctxs in [('closure', closed_ctx), ('fresh line end', fresh_ctx)]:
    for k, p, ids in ctxs[:6]:
        base = lib.run(m, ids[:p + 1])['logits'][0]
        rec = {}
        g = lib.run(m, ids[:p + 1], hooks=[gl.emulation_hook(P, record=rec)])['logits'][0]
        print(f'  {name:14s} {k} tok {p}: score {rec["score"]:.0f} write {rec["write"]:+.3f} | max |dlogit| over vocab {float((g - base).abs().max()):.4f}')
        print(f'       base : {top(base)}')
        print(f'       graft: {top(g)}')
print('\n===== (A) in-loop flip positions, then forced continuation with the graft active =====')
for k, p, ids in loop_ctx:
    base = lib.run(m, ids[:p + 1])['logits'][0]
    rec = {}
    g = lib.run(m, ids[:p + 1], hooks=[gl.emulation_hook(P, record=rec)])['logits'][0]
    print(f'\n  {k} tok {p} (loop token {p - X["data"][k]["P0"] - X["data"][k]["loop_start"]}): ...{dec(ids[p - 14:p + 1])!r}')
    print(f'     score {rec["score"]:.0f} write {rec["write"]:+.1f}')
    print(f'     base : {top(base)}')
    print(f'     graft: {top(g)}')
    nl = lib.ids_of(tok, '\n\n')
    for label, extra in [('after forced </think>', [THINK]), ('after </think> + newlines', [THINK] + nl),
                         ('after ... + The final answer is', [THINK] + nl + lib.ids_of(tok, 'The final answer is'))]:
        full = ids[:p + 1] + extra
        rec = {}
        gg = lib.run(m, full, hooks=[gl.emulation_hook(P, record=rec)])['logits'][0]
        bb = lib.run(m, full)['logits'][0]
        print(f'     {label:34s}: score {rec["score"]:.0f} write {rec["write"]:+.1f} | base: {top(bb, 3)}')
        print(f'     {"":34s}  graft: {top(gg, 3)}')
