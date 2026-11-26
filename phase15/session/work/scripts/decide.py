"""Next-token distributions at repeated decision points of one generation (one forward pass).

Usage: python decide.py <pass_no> <problem_id> <substring> [--max 12] [--minsep 3] [--track t1,t2]
For every occurrence of <substring> in the generated text, the top candidates for the next token
are printed, with the </think> logit.
"""
import sys, torch, argparse, bisect
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('pass_no', type=int); ap.add_argument('pid'); ap.add_argument('needle')
ap.add_argument('--max', type=int, default=12)
ap.add_argument('--minsep', type=int, default=3)
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
ap.add_argument('--track', default='')
ap.add_argument('--last', action='store_true', help='show the last occurrences instead of the first')
a = ap.parse_args()
a.needle = a.needle.encode().decode('unicode_escape')     # allows \x24 for '$' through the shells
m, tok = lib.load(a.model_dir)
prac = {p['id']: p for p in lib.practice()}
g = lib.gens(a.pass_no)[a.pid]
p_ids = lib.prompt_ids(tok, prac[a.pid]['q'])
g_ids = lib.ids_of(tok, g['text'])
ids = p_ids + g_ids
THINK = tok.convert_tokens_to_ids('</think>')
extra = [tok.convert_tokens_to_ids(t) for t in a.track.split(',') if t]
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
full = dec(g_ids)
cum, pos = [], 0
for t in g_ids:
    pos += len(dec([t])); cum.append(pos)
occ, start = [], 0
while True:
    k = full.find(a.needle, start)
    if k < 0:
        break
    start = k + 1
    occ.append(k)
sel = []
for k in occ:
    j = bisect.bisect_left(cum, k + len(a.needle))
    if j >= len(g_ids) - 1:
        break
    if sel and j - sel[-1][1] < a.minsep:
        continue
    sel.append((k, j))
if a.last:
    sel = sel[-a.max:]
else:
    sel = sel[:a.max]
positions = [len(p_ids) + j for _, j in sel]
print(f'{a.pid}: needle {a.needle!r}: {len(occ)} occurrences in the text, {len(sel)} selected')
if not positions:
    sys.exit(0)
r = lib.run(m, ids, positions=positions)
L = r['logits']
for n, (k, j) in enumerate(sel):
    v, i = torch.topk(L[n], 12)
    cands = ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}' for t, x in zip(i, v))
    ex = ' '.join(f'{dec([t])!r}:{float(L[n, t]):.1f}' for t in extra)
    nxt = dec(g_ids[j + 1:j + 16])
    print(f'gen tok {j:6d}: ...{full[max(0,k-25):k+len(a.needle)]!r} -> actual next {nxt!r}')
    print(f'            top: {cands} | </think> {float(L[n, THINK]):.1f} {ex}')
