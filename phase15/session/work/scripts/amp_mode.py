"""Does amplifying the answer-mode feature make the model more decisive at final-phase runaway
positions?  At such positions (pass 3), add dh to the residual at the OUTPUT of layer 14's MLP so
that the answer-mode readout (A on the normalised input, A.x ~ -1 thinking / +1 answer mode)
rises by delta, and look at the next-token distribution.
Residual-space direction a = A / (1 + w_post) reproduces the readout on the raw residual
(x = rms(h) * (1 + w)), so dh = delta / (rstd * |a|^2) * a.

Usage: python amp_mode.py [--n 12]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--maxctx', type=int, default=14000)
a = ap.parse_args()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(3)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
H = m.model.language_model
A = torch.load('/home/user/wsp14/work/out/mode_lda_L14.pt', weights_only=False)['A'].cuda().double()
s_post = (1 + H.layers[14].post_attention_layernorm.weight.double())
a_dir = A / s_post
a2 = float(a_dir @ a_dir)
WAIT = tok(' Wait', add_special_tokens=False)['input_ids'][0]
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    if '</think>' not in g['text'] or (g['finish'] == 'stop' and chr(92) + 'boxed' in g['text']):
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    first = ids.index(THINK)
    for i in range(first + 200, min(len(ids) - 30, a.maxctx)):
        if dec(ids[i:i + 1]) == '\n':
            j = i + 1
            while j < len(ids) - 1 and dec(ids[j:j + 1]) != '' and dec(ids[j:j + 1]).strip() == '':
                j += 1
            if ids[j] == WAIT:
                ctxs.append((pid, j - 1, ids[:j]))
                break
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} final-phase line-start positions where the model wrote " Wait" (the entry into a doubt loop)')


def hook_for(delta, state):
    def pre_layer(mod, args, kwargs):
        state['hnorm'] = float((args[0] if args else kwargs['hidden_states'])[0, -1].double().norm())
    def post_mlp(mod, args, kwargs, out):
        rstd = 32.0 / state['hnorm']
        dh = (delta / (rstd * a2)) * a_dir
        o = out.clone()
        o[0, -1] += dh.to(o.dtype)
        return o
    def pre_next(mod, args, kwargs):
        h = (args[0] if args else kwargs['hidden_states'])[0, -1].double()
        state['readout_after'] = float(a_dir @ h) * 32.0 / float(h.norm())
    return [lambda: H.layers[14].register_forward_pre_hook(pre_layer, with_kwargs=True),
            lambda: H.layers[14].mlp.register_forward_hook(post_mlp, with_kwargs=True),
            lambda: H.layers[15].register_forward_pre_hook(pre_next, with_kwargs=True)]


deltas = [0.0, 1.0, 2.0, 4.0, 8.0]
agg = {d: dict(p_wait=0.0, p_top=0.0, p_think=0.0, ro=0.0) for d in deltas}
examples = {d: [] for d in deltas}
for pid, p, ids in ctxs:
    print(f'  {pid}: ...{dec(ids[-22:])!r}')
    for d in deltas:
        state = {}
        hooks = [] if d == 0.0 else hook_for(d, state)
        lg = lib.run_pos(m, ids, [len(ids) - 1], hooks=hooks)['logits'][0]
        pr = torch.softmax(lg, 0)
        agg[d]['p_wait'] += float(pr[WAIT]); agg[d]['p_top'] += float(pr.max()); agg[d]['p_think'] += float(pr[THINK])
        agg[d]['ro'] += state.get('readout_after', float('nan'))
        v, i = torch.topk(lg, 4)
        examples[d].append(' '.join(f'{dec([int(t)])!r}:{float(x):.0f}' for t, x in zip(i, v)))
    for d in deltas:
        print(f'      delta {d:3.0f}: {examples[d][-1]}')
n = len(ctxs)
print('\nmean over positions: delta | P(" Wait") | P(top) | P(</think>) | readout on the residual after layer 14')
for d in deltas:
    print(f'   {d:3.0f} | {agg[d]["p_wait"]/n:.3f} | {agg[d]["p_top"]/n:.3f} | {agg[d]["p_think"]/n:.3f} | {agg[d]["ro"]/n:+.2f}')
