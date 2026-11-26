"""Verify the two-neuron state (N1 + N4) with real forward passes against the pristine base and
the saved one-neuron state g1.

Usage: python verify_g2.py
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
tok = lib.AutoTokenizer.from_pretrained(lib.MODEL_DIR)
G2 = lib.gens(2)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
m_b, _ = lib.load('/home/user/wsp14/sandbox/base_model')
m_1, _ = lib.load('/home/user/wsp14/work/g1_L14n1439_t425')
m_2, _ = lib.load('/home/user/wsp14/sandbox/model')


def top(L, k=3):
    p = torch.softmax(L, 0)
    v, i = torch.topk(L, k)
    return ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}({float(p[t]):.2f})' for t, x in zip(i, v))


print('== runaway enumerations that never closed in pass 2: base / g1 / g2 at three late positions')
for pid in ['p28', 'p29', 'p31', 'p37', 'p44', 'p57', 'p99', 'p123']:
    g = G2[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    T = len(ids)
    line = f'  {pid}:'
    for back in (6000, 3000, 200):
        p = T - 2 - back
        L = {n: lib.run_pos(mm, ids[:p + 1], [p])['logits'][0] for n, mm in (('base', m_b), ('g1', m_1), ('g2', m_2))}
        pt = {n: float(torch.softmax(v, 0)[THINK]) for n, v in L.items()}
        line += f' | -{back}: P(</think>) base {pt["base"]:.3f} g1 {pt["g1"]:.3f} g2 {pt["g2"]:.3f}'
    print(line)
    p = T - 202
    print(f'       ...{dec(ids[p - 14:p + 1])!r}')
    for n, mm in (('base', m_b), ('g1 ', m_1), ('g2 ', m_2)):
        print(f'       {n}: {top(lib.run_pos(mm, ids[:p + 1], [p])["logits"][0])}')
print('\n== cleanly ended traces: final-answer positions, g2 against g1 (isolates N4) and against base')
tot1 = tot2 = 0.0; n = 0; dis = 0
for pid in sorted(G2, key=lambda s: int(s[1:])):
    g = G2[pid]
    if g['finish'] != 'stop' or '</think>' not in g['text']:
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, g['text'])
    c0 = ids.index(THINK)
    pos = list(range(c0, len(ids) - 1, 25))
    if len(pos) < 3:
        continue
    A = lib.run_pos(m_b, ids, pos)['logits']; B = lib.run_pos(m_1, ids, pos)['logits']; C = lib.run_pos(m_2, ids, pos)['logits']
    d21 = float((C - B).abs().max()); d2b = float((C - A).abs().max())
    tot1 = max(tot1, d21); tot2 = max(tot2, d2b); n += len(pos)
    dis += int((C.argmax(-1) != B.argmax(-1)).sum())
print(f'  {n} final-answer positions over all cleanly ended traces: largest |g2 - g1| logit change {tot1:.2f}; largest |g2 - base| {tot2:.2f}; top-token disagreements g2 vs g1: {dis}')
print('\n== in-loop positions of pass-1 traces (thinking-phase verbatim loops): g2 must still close')
import graftlib as gl
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
shown = 0
for k in sorted(X['data'], key=lambda s: int(s[1:])):
    D = X['data'][k]
    if not D['loop_start'] or int(k[1:]) % 2:
        continue
    ids = D['ids'].tolist()
    if len(ids) > 9000:
        continue
    P = gl.load_params(); P['_m'] = m_b
    act = gl.activation(D['x'][X['layers'].index(14)].float()[D['P0'] + D['loop_start']:], P)
    hit = torch.nonzero(5.5 * act >= 3.5).flatten()
    if not hit.numel():
        continue
    p = D['P0'] + D['loop_start'] + int(hit[0])
    b = lib.run_pos(m_b, ids[:p + 1], [p])['logits'][0]
    e = lib.run_pos(m_2, ids[:p + 1], [p])['logits'][0]
    print(f'  {k}: base {top(b)}   ->   g2 {top(e)}')
    shown += 1
    if shown >= 6:
        break
