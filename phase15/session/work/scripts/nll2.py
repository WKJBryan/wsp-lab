"""Teacher-forced NLL on two text kinds, for a given model directory:
  early  : generated tokens 100..1600 of every pass-1 trace (ordinary reasoning, before any loop)
  preloop: the 1,000 tokens ending 100 tokens before each pass-1 loop start (re-check phase)
Usage: python nll2.py <model_dir> [more dirs...]   (one model loaded at a time)
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
tok = lib.AutoTokenizer.from_pretrained('/home/user/wsp14/sandbox/base_model')
wins = {'early': [], 'preloop': []}
for pid in sorted(G, key=lambda s: int(s[1:])):
    g = G[pid]
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    gen = lib.ids_of(tok, g['text'])
    wins['early'].append(p_ids + gen[100:1600])
    s = loops[pid]['loop_start']
    if s and s > 1200:
        wins['preloop'].append(p_ids + gen[s - 1100:s - 100])
for md in sys.argv[1:]:
    m, _ = lib.load(md)
    out = []
    for name, ws in wins.items():
        tot, cnt = 0.0, 0
        for ids in ws:
            x = torch.tensor([ids], device='cuda')
            lg = m(input_ids=x).logits[0, -1000:].float()
            tgt = x[0, -1000:]
            tot += float(torch.nn.functional.cross_entropy(lg[:-1], tgt[1:], reduction='sum')); cnt += lg.shape[0] - 1
        out.append(f'{name} NLL {tot / cnt:.4f} (n={cnt})')
    print(f'{md.split("/")[-1]:22s} ' + ' | '.join(out), flush=True)
    lib._cache.pop(md, None)
    del m
    torch.cuda.empty_cache()
