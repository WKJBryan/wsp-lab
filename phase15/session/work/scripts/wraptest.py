"""What does the model write after candidate wrap-up starter words at real in-loop line starts?

For held-out traces (even ids): take line-start positions inside the loop (the token right before
the first word of a line), one to three per trace near the start of the loop. For each candidate
starter (forced after the context) report the model's next-token distribution, averaged over the
positions, and the probability of closing the thought after a few full wrap-up phrases.

Usage: python wraptest.py [--per 2] [--maxctx 14000]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--per', type=int, default=1)
ap.add_argument('--maxctx', type=int, default=9000)
ap.add_argument('--maxctxs', type=int, default=24)
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
G = lib.gens(1)
prac = {p['id']: p for p in lib.practice()}
LS = torch.load('/home/user/wsp14/work/out/linestart_1.pt')
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
THINK = tok.convert_tokens_to_ids('</think>')
ids_of = lambda s: tok(s, add_special_tokens=False)['input_ids']
starters = [' Wait', " I'll", ' I', ' I will', ' Therefore', ' Thus', ' The final answer is', ' Final', ' So', ' Okay', ' Let', ' So the final answer is', ' The']
phrases = [" I'll write out the solution now.\n", " I will write the solution.\n", " I'll generate the response.\n", " I will start the response now.\n",
           " Therefore, the final answer is", " Thus, the final answer is", " Let's write out the response.\n"]
print('tokenisations:')
for s in starters + phrases:
    print(f'   {s!r:40s} -> {[dec([t]) for t in ids_of(s)]}')
ctxs = []
for k in sorted(LS, key=lambda s: int(s[1:])):
    if int(k[1:]) % 2:          # held-out = even ids
        continue
    d = LS[k]
    lp = [p for p, r in zip(d['pos'].tolist(), d['region']) if r == 'loop']
    if not lp:
        continue
    P0 = len(lib.prompt_ids(tok, prac[k]['q']))
    lp = [p for p in lp if p - P0 < a.maxctx]
    if not lp:
        continue
    for p in lp[:a.per]:
        if len(ctxs) >= a.maxctxs:
            break
        ids = lib.prompt_ids(tok, prac[k]['q']) + lib.ids_of(tok, G[k]['text'])
        ctxs.append((k, p, ids[:p + 1], ids[p + 1:p + 2]))
print(f'\n{len(ctxs)} in-loop line-start positions from {len(set(c[0] for c in ctxs))} traces')
for name in starters:
    tot = torch.zeros(248320)
    for k, p, ids, nxt in ctxs:
        x = ids + ids_of(name)
        r = lib.run(m, x)
        tot += torch.softmax(r['logits'][0], 0).cpu()
    v, i = torch.topk(tot / len(ctxs), 9)
    print(f'   after {name!r:24s}: ' + ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i, v)))
print('\nP(</think>) after a full phrase (+ as written), averaged over the positions; also mean log-prob of the phrase per token')
for ph in phrases:
    pt, nll, n = 0.0, 0.0, 0
    for k, p, ids, nxt in ctxs:
        tids = ids_of(ph)
        x = ids + tids
        r = lib.run(m, x)
        pt += float(torch.softmax(r['logits'][0], 0)[THINK])
        r2 = lib.run(m, x + ids_of('\n'))       # phrase then newline
        pt2 = float(torch.softmax(r2['logits'][0], 0)[THINK])
        n += 1
    print(f'   {ph!r:40s}: P(</think>) right after {pt/n:.3f}; after an extra newline {pt2/n:.3f}')
