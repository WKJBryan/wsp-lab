"""Premature-closure risk at the START of the thinking: for every pass-1 problem, run the prompt
plus the first 600 generated tokens under each model and report the probability of </think>
(temperature 0.6, as in the protocol) at each position: the maximum and where.

Usage: python earlyrisk.py <model_dir> [more dirs...]
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
THINK = tok_id = 248069
pids = sorted(G, key=lambda s: int(s[1:]))
for md in sys.argv[1:]:
    m, tok = lib.load(md)
    mx, where = [], []
    for pid in pids:
        ids = lib.prompt_ids(tok, prac[pid]['q']) + lib.ids_of(tok, G[pid]['text'])[:600]
        lg = m(input_ids=torch.tensor([ids], device='cuda')).logits[0].float()
        p = torch.softmax(lg / 0.6, -1)[:, THINK].cpu()      # p[i] = P(</think> as token i+1)
        k = int(p.argmax())
        mx.append(float(p[k])); where.append((pid, k + 1 - len(ids) + 600, float(p[k])))
    where.sort(key=lambda t: -t[2])
    print(f'{md.split("/")[-1]:22s} max P_T0.6(</think>) over the first 600 tokens: median over traces {sorted(mx)[64]:.4f} | '
          f'traces with max > 0.01: {sum(1 for x in mx if x > 0.01):3d} | > 0.05: {sum(1 for x in mx if x > 0.05):2d} | > 0.2: {sum(1 for x in mx if x > 0.2):2d}')
    print('    highest: ' + ' '.join(f'{p}@{t}:{v:.3f}' for p, t, v in where[:10]))
    lib._cache.pop(md, None); del m; torch.cuda.empty_cache()
