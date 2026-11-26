"""Profiles of the two detector scores along the traces that never closed in a pass.
Usage: python neverclose.py <pass_no> <pid,pid,...>"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
pass_no = int(sys.argv[1]); pids = sys.argv[2].split(',')
m, tok = lib.load('/home/user/wsp14/sandbox/base_model')
G = lib.gens(pass_no)
prac = {p['id']: p for p in lib.practice()}
G1 = torch.load('/home/user/wsp14/work/out/graft_dirs_L14.pt')
d_ls = G1['d'].cuda()
d_en = torch.load('/home/user/wsp14/work/out/enum_dir_L14.pt', weights_only=False)['d_enum'].cuda()
norm = m.model.language_model.layers[14].post_attention_layernorm
for pid in pids:
    g = G[pid]
    p0 = lib.prompt_ids(tok, prac[pid]['q'])
    ids = p0 + lib.ids_of(tok, g['text'])
    pos = torch.arange(len(p0) + 3, len(ids) - 2, 8).cuda()
    rec = []
    h = norm.register_forward_hook(lambda mod, args, o: rec.append(o[0].float()[pos]))
    try:
        m.model(input_ids=torch.tensor([ids], device='cuda'))
    finally:
        h.remove()
    x = rec[0]
    s1 = (x @ d_ls).cpu(); s4 = (x @ d_en).cpu(); t = (pos.cpu() - len(p0))
    bins = list(range(0, g['n_tokens'], 2000))
    print(f'\n{pid}: {g["n_tokens"]} tokens, finish {g["finish"]}')
    print('  repetition score: median per 2000-token bin: ' + ' '.join(f'{float(s1[(t >= a) & (t < a + 2000)].median()):.0f}' for a in bins[:-1]))
    print('  repetition score: MAX    per 2000-token bin: ' + ' '.join(f'{float(s1[(t >= a) & (t < a + 2000)].max()):.0f}' for a in bins[:-1]))
    print('  enumeration score: median per 2000-token bin: ' + ' '.join(f'{float(s4[(t >= a) & (t < a + 2000)].median()):+.0f}' for a in bins[:-1]))
    print('  enumeration score: MAX    per 2000-token bin: ' + ' '.join(f'{float(s4[(t >= a) & (t < a + 2000)].max()):+.0f}' for a in bins[:-1]))
