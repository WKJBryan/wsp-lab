"""Final-phase loop positions (pass 2): natural probability along the conclusion path after ONE
forced word, using the model's own preferred box format, and what it writes in the box.

Usage: python fp_concl.py [--n 30] [--maxctx 16000]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--maxctx', type=int, default=16000)
ap.add_argument('--model_dir', default='/home/user/wsp14/sandbox/base_model')
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
G = {json.loads(l)['id']: json.loads(l) for l in open('/home/user/wsp14/work/out/gen_2.jsonl', encoding='utf-8')}
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + T(g['text'])
    if THINK not in ids:
        continue
    first = ids.index(THINK)
    later = [i for i in range(first + 20, min(len(ids) - 40, a.maxctx)) if ids[i] == THINK]
    for i in later[1:2]:
        ctxs.append((pid, i - 1, ids[:i], g['ok_boxed']))
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} final-phase loop positions (one per trace)')
# path: " Therefore, the final answer is $\boxed{"   (tokens as the model writes them)
phrase = T(' Therefore, the final answer is $\\boxed{')
print('path tokens:', [dec([t]) for t in phrase])
probs = torch.zeros(len(phrase) - 1)
rows = []
for pid, p, ids, ok in ctxs:
    full = ids + phrase
    # position len(ids)+j has input token phrase[j] and predicts the next token
    pos = list(range(len(ids), len(full)))
    L = lib.run_pos(m, full, pos)['logits']            # [len(phrase), V]
    pr = torch.softmax(L, -1)
    pt = torch.tensor([float(pr[k, phrase[k + 1]]) for k in range(len(phrase) - 1)])
    probs += pt
    v, i = torch.topk(L[-1], 4)                         # what goes inside the box
    after = ' '.join(f'{dec([int(t)])!r}:{float(x):.1f}' for t, x in zip(i, v))
    v0, i0 = torch.topk(L[0], 4)                        # the step right after the forced word
    alt0 = ' '.join(f'{dec([int(t)])!r}:{float(pr[0, t]):.2f}' for t, x in zip(i0, v0))
    rows.append((float(pt[1:].prod()), pid, after, alt0))
    print(f'  {pid:4s} P(path after " Therefore") {float(pt[1:].prod()):.2f} | after "Therefore": {alt0} | after "$\\boxed{{": {after}   (gold {prac[pid]["gold"]!r})')
mean = probs / len(ctxs)
print('\nmean natural P of each path token (after the forced " Therefore"): ' + ' '.join(f'{dec([phrase[k+1]])!r}:{float(mean[k]):.2f}' for k in range(len(mean))))
print(f'mean product after the first word: {float(mean.prod()):.3f}; per-trace product: median {sorted(r[0] for r in rows)[len(rows)//2]:.2f}, frac >= 0.3: {sum(1 for r in rows if r[0] >= 0.3)/len(rows):.2f}')
