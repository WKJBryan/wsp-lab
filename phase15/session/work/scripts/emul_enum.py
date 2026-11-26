"""Emulate graft N1 (verbatim repetition) + N4 (runaway enumeration) with runtime hooks.

(A) the 8 pass-2 traces that never closed (runaway enumerations): logits at positions 6,000,
    3,000 and 200 tokens before the end, base vs grafted.
(B) cleanly ended traces: every 40th position of the final-answer phase must be unchanged.

Usage: python emul_enum.py
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(2)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
G1 = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
v = gl.unit_output_dir(m, THINK)
P1 = dict(layer=14, d=G1['d'], w=G1['w'], theta=425, s=20, c=5.5, v=v, _m=m)
d_en = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum']
P4 = dict(layer=14, d=d_en, w=G1['w'], theta=5, s=10, c=2.0, v=v, _m=m)
hooks = [gl.emulation_hook(P1), gl.emulation_hook(P4)]


def top(L, k=3):
    p = torch.softmax(L, 0)
    v, i = torch.topk(L, k)
    return ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}({float(p[t]):.2f})' for t, x in zip(i, v)) + f' [</think> {float(L[THINK]):.1f} p={float(p[THINK]):.3f}]'


print('== (A) never-closed runaway enumerations (pass 2): base vs N1+N4 at three late positions')
for pid in ['p28', 'p29', 'p31', 'p37', 'p44', 'p57', 'p99', 'p123']:
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    T = len(ids)
    line = f'  {pid}:'
    for back in (6000, 3000, 200):
        p = T - 2 - back
        b = lib.run_pos(m, ids[:p + 1], [p])['logits'][0]
        rec = {}
        h = [gl.emulation_hook(P4, record=rec)]
        e = lib.run_pos(m, ids[:p + 1], [p], hooks=h)['logits'][0]
        pb, pe = torch.softmax(b, 0)[THINK], torch.softmax(e, 0)[THINK]
        line += f'  | -{back}: enum score {rec["score"]:+.0f}, P(</think>) base {float(pb):.3f} -> N4 {float(pe):.3f}'
    print(line)
    p = T - 202
    b = lib.run_pos(m, ids[:p + 1], [p])['logits'][0]
    e = lib.run_pos(m, ids[:p + 1], [p], hooks=hooks)['logits'][0]
    print(f'       ...{dec(ids[p - 20:p + 1])!r}')
    print(f'       base : {top(b)}')
    print(f'       N1+N4: {top(e)}')
    # and what follows the forced </think>
    e2 = lib.run_pos(m, ids[:p + 1] + [THINK] + lib.ids_of(tok, '\n\n'), [p + 2], hooks=hooks)['logits'][0]
    print(f'       N1+N4, after </think> + newlines: {top(e2)}')
print('\n== (B) cleanly ended traces: final-answer phase positions must be unchanged')
mx = 0.0; n = 0
for pid in sorted(G, key=lambda s: int(s[1:]))[:100:5]:
    g = G[pid]
    if g['finish'] != 'stop' or '</think>' not in g['text']:
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    c0 = ids.index(THINK)
    pos = list(range(c0, len(ids) - 1, 40))
    b = lib.run_pos(m, ids, pos)['logits']
    e = lib.run_pos(m, ids, pos, hooks=hooks)['logits']
    d = float((b - e).abs().max()); mx = max(mx, d); n += len(pos)
    agree = float((b.argmax(-1) == e.argmax(-1)).float().mean())
    print(f'  {pid}: {len(pos)} final-phase positions, max |dlogit| {d:.3f}, argmax agreement {agree:.3f}')
print(f'  checked {n} positions; largest logit change anywhere {mx:.3f}')
