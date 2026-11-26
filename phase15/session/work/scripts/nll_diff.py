"""Per-trace NLL difference between two model directories on early windows (tokens 100..1600 of
pass-1 traces). Lists the traces that account for the change.
Usage: python nll_diff.py <model_a> <model_b>"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
A, B = sys.argv[1], sys.argv[2]
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
tok = lib.AutoTokenizer.from_pretrained('/home/user/wsp14/sandbox/base_model')
ws = {}
for pid in sorted(G, key=lambda s: int(s[1:])):
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    ws[pid] = p_ids + lib.ids_of(tok, G[pid]['text'])[100:1600]
res = {}
for name, md in (('A', A), ('B', B)):
    m, _ = lib.load(md)
    for pid, ids in ws.items():
        x = torch.tensor([ids], device='cuda')
        lg = m(input_ids=x).logits[0].float()
        nll = torch.nn.functional.cross_entropy(lg[:-1], x[0, 1:], reduction='none')   # [T-1]
        res.setdefault(pid, {})[name] = nll.cpu()
    lib._cache.pop(md, None); del m; torch.cuda.empty_cache()
rows = []
for pid, d in res.items():
    diff = d['B'] - d['A']                      # index i = prediction of token i+1 (absolute index i+1 in the sequence)
    tot = float(diff.sum())
    # positions of the largest increases, in generated-token index
    v, i = torch.topk(diff, 3)
    rows.append((tot, pid, [(int(j) + 1 - len(lib.prompt_ids(tok, prac[pid]['q'])) + 100, round(float(x), 2)) for j, x in zip(i, v)]))
rows.sort(reverse=True)
print(f'total NLL change B-A over all early windows: {sum(r[0] for r in rows):.1f} nats')
print('traces with the largest change (total nats, generated-token index and size of the 3 largest position increases):')
for tot, pid, top in rows[:12]:
    if tot > 1:
        print(f'  {pid}: {tot:+.1f}  {top}')
print('number of traces with total change > 1 nat:', sum(1 for r in rows if r[0] > 1), '| > 0.1:', sum(1 for r in rows if r[0] > 0.1))
