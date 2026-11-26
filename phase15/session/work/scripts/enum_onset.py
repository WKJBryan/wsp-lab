"""Enumeration-direction score along whole runaway traces (pass 2), in 1500-token bins from the
start of the generation, to see how fast the score builds after the list starts.

Usage: python enum_onset.py
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
torch.set_grad_enabled(False)
de = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum'].cuda()
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(2)
norm = m.model.language_model.layers[14].post_attention_layernorm
dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
for pid in ['p44', 'p123', 'p28', 'p29', 'p99', 'p57']:
    g = G[pid]
    p0 = lib.prompt_ids(tok, prac[pid]['q'])
    gid = lib.ids_of(tok, g['text'])
    ids = p0 + gid
    T = len(ids)
    pos = torch.arange(len(p0) + 3, T - 2, 8).cuda()
    rec = []
    h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[pos]))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    sc = (rec[0] @ de).cpu(); t = (pos.cpu() - len(p0))
    bins = list(range(0, g['n_tokens'], 1500))
    prof = ' '.join(f'{float(sc[(t >= a) & (t < a + 1500)].median()):+.0f}' for a in bins[:-1])
    print(f'\n{pid}: {g["n_tokens"]} tokens; median enumeration score per 1500-token bin from the start:\n   {prof}')
    # crude onset: the first generated token after which the next 400 tokens contain >= 8 newlines with similar line starts
    txt = g['text']
    import re
    lines = txt.split('\n')
    best, onset = 0, None
    for i in range(0, max(0, len(lines) - 40), 5):
        window = [re.sub(r'[-\d]+', '#', ln.strip())[:40] for ln in lines[i:i + 40]]
        c = max(window.count(w) for w in set(window))
        if c >= 25:
            onset = i; break
    if onset is not None:
        chars = sum(len(ln) + 1 for ln in lines[:onset])
        tok_onset = int(chars / len(txt) * g['n_tokens'])
        print(f'   first window of 40 lines with >= 25 same-template lines begins at generated token ~{tok_onset}:  '
              + ' | '.join(repr(ln.strip()[:60]) for ln in lines[onset:onset + 3]))
        near = sc[(t >= tok_onset - 1500) & (t < tok_onset + 1500)]
        print(f'   score around the onset (+/-1500 tokens): median {float(near.median()):+.0f}, max {float(near.max()):+.0f}')
