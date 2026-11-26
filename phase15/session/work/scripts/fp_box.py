"""Final-phase loop positions (pass 2; the graft fired again there): force a backslash token and
read the model's natural continuation. One forward pass per path.

Usage: python fp_box.py [--n 24] [--maxctx 16000]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=24)
ap.add_argument('--maxctx', type=int, default=16000)
ap.add_argument('--model_dir', default='/home/user/wsp14/sandbox/base_model')
a = ap.parse_args()
m, tok = lib.load(a.model_dir)
G = {json.loads(l)['id']: json.loads(l) for l in open('/home/user/wsp14/work/out/gen_2.jsonl', encoding='utf-8')}
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
cands = {' \\\\ (1088)': T(' \\'), '\\\\ (59)': T('\\'), ' $\\\\ (56749)': T('$\\')}
paths = {'boxed {': T('boxed{'), 'boxed{7}': T('boxed{7}')}
tokmap = {k: v[0] for k, v in cands.items()}
BOXED, BRACE = T('boxed')[0], T('{')[0]
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    ids = lib.prompt_ids(tok, prac[pid]['q']) + T(g['text'])
    if THINK not in ids:
        continue
    first = ids.index(THINK)
    later = [i for i in range(first + 20, min(len(ids) - 40, a.maxctx)) if ids[i] == THINK]
    for i in later[1:2] + later[len(later) // 2: len(later) // 2 + 1]:
        ctxs.append((pid, i - 1, ids[:i]))
    if len(ctxs) >= a.n:
        break
print(f'{len(ctxs)} final-phase loop positions from {len(set(c[0] for c in ctxs))} traces')
for pid, p, ids in ctxs[:6]:
    print(f'   {pid}: ...{dec(ids[-16:])!r}')
for cname, ctok in cands.items():
    p_box, p_after, examples = [], [], []
    for pid, p, ids in ctxs:
        full = ids + ctok
        L0 = lib.run_pos(m, full, [len(full) - 1])['logits'][0]
        pr0 = torch.softmax(L0, 0)
        p_box.append(float(pr0[BOXED]))
        full2 = full + [BOXED, BRACE]
        L2 = lib.run_pos(m, full2, [len(full2) - 1])['logits'][0]
        pr2 = torch.softmax(L2, 0)
        v, i = torch.topk(L2, 5)
        p_after.append(float(pr2[T('7')[0]]))       # placeholder, replaced below by top-1 prob
        p_after[-1] = float(v[0])
        examples.append((pid, ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i, v)), float(v[0])))
    print(f'\n  forced {cname}: natural P("boxed" | backslash) mean {sum(p_box)/len(p_box):.2f}, median {sorted(p_box)[len(p_box)//2]:.2f}; '
          f'P(top token after "boxed{{") mean {sum(p_after)/len(p_after):.2f}')
    for pid, ex, v0 in examples[:8]:
        print(f'      {pid}: after "boxed{{": {ex}')
# what follows naturally after a forced backslash at these positions (top tokens)
print('\n  natural top-5 after the forced backslash (first 10 positions):')
for cname, ctok in list(cands.items())[:1]:
    for pid, p, ids in ctxs[:10]:
        L0 = lib.run_pos(m, ids + ctok, [len(ids)])['logits'][0]
        v, i = torch.topk(L0, 5)
        print(f'      {pid}: ' + ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i, v)))
