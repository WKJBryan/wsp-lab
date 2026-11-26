"""CPU: how does the repetition score at a line start grow with the number of earlier verbatim
occurrences of that line?  (pass-1 traces, x-space of layer 14)

Usage: python repcurve.py
"""
import sys, torch, json, collections, statistics
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib, graftlib as gl
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(lib.MODEL_DIR)
X = torch.load('/home/user/wsp14/work/out/xpos_1.pt', weights_only=False)
li = X['layers'].index(14)
G1 = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
P = dict(d=G1['d'], w=G1['w'], theta=0, s=1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
SPACE = {'Ġ', 'Ċ', 'ĠĠ', 'ĠĠĠ', 'ĠĠĠĠ', 'ĠĠĠĠĠĠ', 'ĠĠĠĠĠĠĠĠ', 'ĠĠĠĠĠĠĠĠĠĠ', 'ĠĠĠĠĠĠĠĠĠĠĠĠ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ', '*', '-', 'ĠĠĠĠĊ', '*ĠĠ', 'Ġ*'}
lines_by = collections.defaultdict(list)      # occurrence index -> list of (score, first-occurrence-score)
first_of = {}
for k in sorted(X['data'], key=lambda s: int(s[1:])):
    D = X['data'][k]
    ids = D['ids'].tolist()
    x = D['x'][li].float()
    T = x.shape[0]
    s = (x @ G1['d']).tolist()
    P0 = D['P0']
    end = min(T, P0 + (loops[k]['loop_start'] or 10**9) + 300)
    # line starts: first non-space token after a newline
    starts = []
    i = P0 + 1
    while i < end - 14:
        if 'Ċ' in tok.convert_ids_to_tokens([ids[i]])[0]:
            j = i + 1
            while j < end and tok.convert_ids_to_tokens([ids[j]])[0] in SPACE:
                j += 1
            if j < end - 14:
                starts.append(j)          # index of the first real token of the line
            i = j
        else:
            i += 1
    seen = {}
    for j in starts:
        key = tuple(ids[j:j + 10])
        n = seen.get(key, 0)
        seen[key] = n + 1
        sc = s[j - 1]                      # score at the position predicting the first token of the line
        if n == 0:
            first_of[(k, j)] = sc
        occ = min(n, 9)
        lines_by[occ].append(sc)
print('repetition score at a line start, by number of EARLIER verbatim occurrences of that line (first 10 tokens):')
print('  earlier occurrences | n | median | p25 | p75 | p90')
for n in range(10):
    v = sorted(lines_by[n])
    if not v:
        continue
    q = lambda p: v[int(p * (len(v) - 1))]
    print(f'  {n:2d} | {len(v):6d} | {q(0.5):6.0f} | {q(0.25):6.0f} | {q(0.75):6.0f} | {q(0.9):6.0f}')
# share of positions by score band, for first occurrences vs repeats
def band(v):
    c = collections.Counter()
    for s in v:
        c['<0' if s < 0 else '0-150' if s < 150 else '150-250' if s < 250 else '250-320' if s < 320 else '320-400' if s < 400 else '>=400'] += 1
    return {k: round(c[k] / len(v), 3) for k in ['<0', '0-150', '150-250', '250-320', '320-400', '>=400']}
for n in (0, 1, 2, 3, 5):
    if n in lines_by:
        print(f'  score bands, {n} earlier occurrence(s): {band(lines_by[n])}')
