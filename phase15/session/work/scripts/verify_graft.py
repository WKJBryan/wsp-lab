"""Verify the edited checkpoint against the pristine base with real forward passes (no hooks).

Compares logits at in-loop flip positions, natural closure positions, fresh line ends and the
problem statement, between base_model and the working copy.

Usage: python verify_graft.py [--n 10]
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--edited', default='/home/user/wsp14/sandbox/model')
a = ap.parse_args()
tok = lib.AutoTokenizer.from_pretrained(lib.MODEL_DIR)
m_ed, _ = lib.load(a.edited)
m_ba, _ = lib.load('/home/user/wsp14/sandbox/base_model')
THINK = tok.convert_tokens_to_ids('</think>')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)


def top(L, k=3):
    p = torch.softmax(L, 0)
    v, i = torch.topk(L, k)
    return ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}({float(p[t]):.2f})' for t, x in zip(i, v)) + f' [</think> {float(L[THINK]):.1f}]'


def compare(ids, label):
    b = lib.run(m_ba, ids)['logits'][0]
    e = lib.run(m_ed, ids)['logits'][0]
    print(f'  {label}: max |dlogit| {float((e - b).abs().max()):.4f}')
    print(f'       base  : {top(b)}')
    print(f'       edited: {top(e)}')


loop_ctx, closed_ctx, fresh_ctx = [], [], []
for k in sorted(X['data'], key=lambda s: int(s[1:])):
    if int(k[1:]) % 2:
        continue
    D = X['data'][k]
    ids = D['ids'].tolist()
    if D['loop_start'] and len(loop_ctx) < a.n:
        P = gl.load_params()
        P['_m'] = m_ba
        act = gl.activation(D['x'][X['layers'].index(14)].float()[D['P0'] + D['loop_start']:], P)
        hit = torch.nonzero(5.5 * act >= 3.5).flatten()
        if hit.numel() and len(ids) <= 11000:
            loop_ctx.append((k, D['P0'] + D['loop_start'] + int(hit[0]), ids))
    elif D['finish'] == 'stop' and THINK in ids:
        closed_ctx.append((k, ids.index(THINK) - 1, ids))
        ends = [jj for jj in range(D['P0'] + 400, ids.index(THINK) - 200) if dec(ids[jj + 1:jj + 2]) == '\n']
        if ends:
            fresh_ctx.append((k, ends[len(ends) // 2], ids))
probs = []
for k in sorted(X['data'], key=lambda s: int(s[1:])):
    if int(k[1:]) % 2:
        continue
    probs.append((k, X['data'][k]['P0'], X['data'][k]['ids'].tolist()))
print(f'\n== problem statements (must be unchanged)')
for k, n0, ids in probs[:6]:
    compare(ids[:n0], k)
print('\n== natural closure positions (must be unchanged)')
for k, p, ids in closed_ctx:
    compare(ids[:p + 1], f'{k} tok {p}')
print('\n== fresh line ends (must be unchanged)')
for k, p, ids in fresh_ctx:
    compare(ids[:p + 1], f'{k} tok {p}')
print('\n== in-loop flip positions (edited model should now prefer </think>)')
for k, p, ids in loop_ctx:
    compare(ids[:p + 1], f'{k} tok {p}')
    e = lib.run(m_ed, ids[:p + 1] + [THINK] + lib.ids_of(tok, '\n\n'))['logits'][0]
    print(f'       edited, after </think> + newlines: {top(e)}')
