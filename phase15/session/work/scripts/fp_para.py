"""(1) Enumeration-direction scores in the final phase of traces that ENDED cleanly vs those that
ran to the cap.  (2) At final-phase runaway positions: force a paragraph break and read the natural
continuation along the canonical ending "\\n\\n \\boxed {".

Usage: python fp_para.py [--n 24]
"""
import sys, torch, argparse, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=24)
ap.add_argument('--maxctx', type=int, default=16000)
a = ap.parse_args()
de = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum']
M = torch.load('/home/user/wsp14/work/out/mode_2_L14.pt', weights_only=False)['data']
stop_max, cap_max = [], []
for k, d in M.items():
    ph = d['phase'] == 1
    if not ph.any():
        continue
    v = float((d['x'].float()[ph] @ de).max())
    (stop_max if d['finish'] == 'stop' else cap_max).append((v, k, d['ok']))
print(f'enumeration-direction max over the FINAL phase per trace:')
print(f'  ended cleanly ({len(stop_max)} traces): ' + ' '.join(f'{k}:{v:+.0f}' for v, k, ok in sorted(stop_max, reverse=True)[:12]))
print(f'  ran to cap   ({len(cap_max)} traces): ' + ' '.join(f'{k}:{v:+.0f}' for v, k, ok in sorted(cap_max, reverse=True)[:8]))
print(f'  ended cleanly: traces with max >= +15: {sum(1 for v, k, ok in stop_max if v >= 15)}, >= +25: {sum(1 for v, k, ok in stop_max if v >= 25)} | capped: >= +15: {sum(1 for v, k, ok in cap_max if v >= 15)}/{len(cap_max)}')
# ---------- (2)
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(2)
prac = {p['id']: p for p in lib.practice()}
THINK = tok.convert_tokens_to_ids('</think>')
T = lambda s: tok(s, add_special_tokens=False)['input_ids']
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
path = T('\n\n\\boxed{')
print('\ncanonical-ending path tokens after the line end:', [dec([t]) for t in path])
ctxs = []
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    if g['finish'] != 'length' or '</think>' not in g['text']:
        continue
    ids = lib.prompt_ids(tok, prac[pid]['q']) + T(g['text'])
    first = ids.index(THINK)
    later = [i for i in range(first + 20, min(len(ids) - 40, a.maxctx)) if ids[i] == THINK]
    for i in later[1:2]:
        ctxs.append((pid, i - 1, ids[:i]))
    if len(ctxs) >= a.n:
        break
acc = torch.zeros(len(path) - 1); prod = []; shown = 0
for pid, p, ids in ctxs:
    full = ids + path
    pos = list(range(len(ids), len(full)))
    L = lib.run_pos(m, full, pos)['logits']
    pr = torch.softmax(L, -1)
    pt = torch.tensor([float(pr[k, path[k + 1]]) for k in range(len(path) - 1)])
    acc += pt
    prod.append(float(pt.prod()))
    if shown < 6:
        v0, i0 = torch.topk(L[0], 5)
        v1, i1 = torch.topk(L[1], 4)
        v2, i2 = torch.topk(L[2], 4)
        print(f'  {pid}: ...{dec(ids[-10:])!r} + "\\n\\n" -> ' + ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i0, torch.softmax(L[0], -1)[i0]))
              + f'\n        then after "\\n\\n\\": ' + ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i1, torch.softmax(L[1], -1)[i1]))
              + f'\n        then after "...\\boxed": ' + ' '.join(f'{dec([int(t)])!r}:{float(x):.2f}' for t, x in zip(i2, torch.softmax(L[2], -1)[i2])))
        shown += 1
mean = acc / len(ctxs)
print(f'\nat {len(ctxs)} final-phase runaway positions (line ends): natural P of each canonical token after a forced paragraph break: '
      + ' '.join(f'{dec([path[k+1]])!r}:{float(mean[k]):.2f}' for k in range(len(mean))))
print(f'per-trace product along the path: median {sorted(prod)[len(prod)//2]:.3f}, share >= 0.25: {sum(1 for x in prod if x >= 0.25)/len(prod):.2f}, >= 0.5: {sum(1 for x in prod if x >= 0.5)/len(prod):.2f}')
