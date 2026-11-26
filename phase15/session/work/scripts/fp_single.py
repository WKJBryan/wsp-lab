"""At final-phase runaway positions (pass 3): which SINGLE forced token is naturally followed by
"\\boxed{" ?  For each (first token, continuation) pair one forward pass per context gives the
natural probability of every continuation token.

Usage: python fp_single.py [--n 16] [--maxctx 16000]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=16)
ap.add_argument('--maxctx', type=int, default=16000)
a = ap.parse_args()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(3)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
B = T('\\boxed{')
pairs = [
    (':**', '\n\\boxed{'), (':**', '\n\n\\boxed{'), ('**', 'Answer:**\n\\boxed{'), ('Answer', ':**\n\\boxed{'),
    (' \\(', '\\boxed{'), (' \\(', ' \boxed{'), (' \\(', '\n\\boxed{'),
    (' $$', '\n\\boxed{'), (' $$', '\\boxed{'), (' $$', ' \boxed{'),
    ('$$', '\n\\boxed{'), ('$$', '\\boxed{'), ('$$', '\n\n\\boxed{'),
    (' \\[', '\\boxed{'), (' \\[', '\n\\boxed{'),
    (' $', '\\boxed{'), (' $', '\n\\boxed{'), (' \\', 'boxed{'),
    ('The final', ' answer is \\boxed{'), (' Therefore', ', the answer is \\boxed{'), (' So', ' the answer is \\boxed{'),
]
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    if '</think>' not in g['text'] or (g['finish'] == 'stop' and chr(92) + 'boxed' in g['text']):
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + T(g['text'])
    first = ids.index(THINK)
    ends = [i for i in range(first + 300, min(len(ids) - 40, a.maxctx)) if dec(ids[i + 1:i + 2]) == '\n']
    if not ends:
        continue
    ctxs.append((pid, ends[len(ends) // 3], ids[:ends[len(ends) // 3] + 1]))
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} final-phase line-end positions from final-phase runaways (pass 3): ' + ' '.join(c[0] for c in ctxs))
res = {}
for fname, pname in pairs:
    seq = T(fname) + T(pname)
    acc = torch.zeros(len(seq) - 1); prods = []
    for pid, p, ids in ctxs:
        full = ids + seq
        pos = list(range(len(ids), len(full)))
        L = lib.run_pos(m, full, pos)['logits']
        pr = torch.softmax(L, -1)
        pt = torch.tensor([float(pr[k, seq[k + 1]]) for k in range(len(seq) - 1)])
        acc += pt; prods.append(float(pt.prod()))
    res[(fname, pname)] = (float((acc / len(ctxs)).prod()), sorted(prods)[len(prods) // 2], sum(1 for x in prods if x >= 0.25) / len(prods), acc / len(ctxs))
print('\nforced first token + continuation: natural P of each continuation token | mean-product | median per-trace product | share of traces >= 0.25')
for (fname, pname), (mp, med, share, mean) in sorted(res.items(), key=lambda kv: -kv[1][0]):
    print(f'  {fname!r:13s} + {pname:26s}: ' + ' '.join(f'{float(x):.2f}' for x in mean) + f' | {mp:.3f} | {med:.3f} | {share:.2f}')
