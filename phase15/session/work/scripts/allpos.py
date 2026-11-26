"""Collect the residual stream at ALL token positions for a subset of traces (several layers).

Per trace the context is: prompt + generation up to (loop_start + 800) tokens for looping traces,
the whole text for traces that closed, the first 6000 tokens for non-periodic cut-off traces.
Saved per trace: resid [n_layers, T, 1024] (fp16 to save space), the line-start mask, P0, loop_start.

Usage: python allpos.py [--every 2] [--layers 10,12,13,14,15,16,18]
"""
import sys, torch, argparse, json, random, re
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--pass', dest='pass_no', type=int, default=1)
ap.add_argument('--layers', default='10,12,13,14,15,16,18')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--save', default='/home/user/wsp14/work/out/allpos_1.pt')
ap.add_argument('--offset', type=int, default=0)
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(',')]
m, tok = lib.load()
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(a.pass_no)
loops = json.load(open(f'/home/user/wsp14/work/out/loops_{a.pass_no}.json'))
dec = lambda t: tok.decode([t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
SPACE = re.compile(r'^[\s\*\-\u2022]*$')


def line_start_mask(ids):
    ls = torch.zeros(len(ids), dtype=torch.bool)
    i = 0
    while i < len(ids):
        if '\n' in dec(ids[i]):
            j = i + 1
            while j < len(ids) and SPACE.match(dec(ids[j])):
                j += 1
            if j < len(ids):
                ls[j - 1] = True
            i = j
        else:
            i += 1
    return ls


pids = sorted(G, key=lambda s: int(s[1:]))[a.offset::a.every]
data = {}
for n, pid in enumerate(pids):
    g = G[pid]
    p_ids = lib.prompt_ids(tok, prac[pid]['q'])
    ids = p_ids + lib.ids_of(tok, g['text'])
    P0 = len(p_ids)
    s = loops[pid]['loop_start']
    if s:
        ids = ids[:min(len(ids), P0 + s + 814)]
    elif g['finish'] == 'stop':
        ids = ids[:len(ids)]
    else:
        ids = ids[:min(len(ids), P0 + 6000)]
    x = torch.tensor([ids], device='cuda')
    rec = []
    handles = []
    for layer_idx in layers:
        module = m.model.language_model.layers[layer_idx]
        def pre(mod, args, kwargs):
            rec.append((args[0] if args else kwargs['hidden_states'])[0].half().cpu())
        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        m.model(input_ids=x)
    finally:
        for h in handles:
            h.remove()
    data[pid] = dict(resid=torch.stack(rec), ls=line_start_mask(ids), P0=P0, loop_start=s, finish=g['finish'])
    if n % 12 == 0:
        print(f'  {n}/{len(pids)} traces; last T={len(ids)} loop_start={s}', flush=True)
torch.save(dict(layers=layers, data=data), a.save)
print('saved', a.save, 'traces', len(data))
