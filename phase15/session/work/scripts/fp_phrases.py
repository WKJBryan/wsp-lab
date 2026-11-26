"""In the FINAL-ANSWER phase (after the first </think>), at positions where the model fell back into
a loop (the graft fired again: the next token is another </think>): what does the model write
naturally after one forced candidate word?  One forward pass per path gives the natural
probability of every token along a teacher-forced phrase.

Usage: python fp_phrases.py [--n 24] [--maxctx 20000]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=24)
ap.add_argument('--maxctx', type=int, default=20000)
ap.add_argument('--model_dir', default='/home/user/wsp14/sandbox/base_model')
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
G = {json.loads(l)['id']: json.loads(l) for l in open('/home/user/wsp14/work/out/gen_2.jsonl', encoding='utf-8')}
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ids_of = lambda s: tok(s, add_special_tokens=False)['input_ids']
cands = {
    ' Therefore': ['ĠTherefore'],
    ' The': ['ĠThe'],
    ' So': ['ĠSo'],
    ' \\(': ['Ġ\\('],
}
paths = {
    'A ":, the final answer is \\(\\boxed': [',', 'Ġthe', 'Ġfinal', 'Ġanswer', 'Ġis', 'Ġ\\(', '\\', 'ox'],
    'B ":, the answer is \\(\\boxed': [',', 'Ġthe', 'Ġanswer', 'Ġis', 'Ġ\\(', '\\', 'ox'],
    'C " is \\(\\boxed': ['Ġis', 'Ġ\\(', '\\', 'ox'],
}
tokmap = {}
for t in [',', 'Ġthe', 'Ġfinal', 'Ġanswer', 'Ġis', 'Ġ\\(', '\\', 'ox', 'ed', '{'] + [x[0] for x in cands.values()]:
    tokmap[t] = tok.convert_tokens_to_ids(t)
    assert tokmap[t] is not None and tokmap[t] != tok.unk_token_id, t
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + ids_of(g['text'])
    first = ids.index(THINK) if THINK in ids else None
    if first is None:
        continue
    later = [i for i in range(first + 20, len(ids) - 40) if ids[i] == THINK and i < a.maxctx]
    for i in later[1:2] + later[len(later) // 2: len(later) // 2 + 1]:      # the position predicting </think>
        ctxs.append((pid, i - 1, ids[:i]))
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} final-phase loop positions from {len(set(c[0] for c in ctxs))} traces')
# natural picture at the firing positions (base model, no graft)
nl_think, tops = [], {}
for pid, p, ids in ctxs:
    L = lib.run_pos(m, ids, [len(ids) - 1])['logits'][0]
    pr = torch.softmax(L, 0)
    nl_think.append(float(L[THINK]))
    for name, (t0,) in cands.items():
        tops.setdefault(name, []).append(float(L[tokmap[t0]]))
print(f'natural </think> logit at these positions: mean {sum(nl_think)/len(nl_think):.1f}; ' + ' '.join(f'{k}: {sum(v)/len(v):.1f}' for k, v in tops.items()))
for cname, ctoks in cands.items():
    for pname, ptoks in paths.items():
        seq = [tokmap[t] for t in ctoks] + [tokmap[t] for t in ptoks]
        n_pos = len(seq)
        acc = torch.zeros(n_pos); alt = [[] for _ in range(n_pos)]; cnt = 0
        for pid, p, ids in ctxs:
            full = ids + seq
            pos = list(range(len(ids) - 1, len(ids) - 1 + n_pos))
            L = lib.run_pos(m, full, pos)['logits']
            pr = torch.softmax(L, 0)
            acc += torch.tensor([float(pr[k, seq[k]]) for k in range(n_pos)])
            for k in range(n_pos):
                v, i = torch.topk(L[k], 2)
                alt[k].append(f'{dec([int(i[0])])!r}:{float(v[0]):.0f}/{dec([int(i[1])])!r}:{float(v[1]):.0f}')
            cnt += 1
        mean = acc / cnt
        prod = float(mean.prod())
        print(f'\n  forced {cname!r} then path {pname}')
        print('    mean natural P of each path token: ' + ' '.join(f'{t}:{float(x):.2f}' for t, x in zip(ptoks, mean[1:])) + f'  | first token {float(mean[0]):.2f} | product {prod:.3f}')
        print('    typical top-2 at steps (last position): ' + ' | '.join(alt[k][-1] for k in range(0, n_pos, max(1, n_pos // 5))))
