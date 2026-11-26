"""Premature-closure hazard along the base model's own trajectories (pass 1).

For each trace: p_i = P_T0.6(</think>) at every position up to the base loop start (+300 tokens).
  forced  = first position with p_i >= 0.5 (the intended closure at strong repetition)
  hazard  = sum of p_i over all EARLIER positions = expected number of spurious closures
            that sampling would produce before the intended one
  n>=0.05 = number of earlier positions with p_i >= 0.05
Usage: python hazard.py <model_dir> [more dirs...]
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(1)
loops = json.load(open('/home/user/wsp14/work/out/loops_1.json'))
THINK = 248069
pids = sorted(G, key=lambda s: int(s[1:]))
for md in sys.argv[1:]:
    m, tok = lib.load(md)
    rows = []
    for pid in pids:
        s = loops[pid]['loop_start']
        n = (s + 300) if s else min(G[pid]['n_tokens'], 8000)
        p_ids = lib.prompt_ids(tok, prac[pid]['q'])
        ids = p_ids + lib.ids_of(tok, G[pid]['text'])[:n]
        L = len(ids)
        ps = []
        for a in range(len(p_ids) - 1, L - 1, 4096):          # chunks of positions
            b = min(L - 1, a + 4096)
            lg = m(input_ids=torch.tensor([ids[:b + 1]], device='cuda')).logits[0, a:b].float()
            ps.append(torch.softmax(lg / 0.6, -1)[:, THINK].cpu())
        p = torch.cat(ps)                                      # p[i] for position a+i (predicting next token)
        forced = torch.nonzero(p >= 0.5).flatten()
        f = int(forced[0]) if forced.numel() else p.numel()
        before = p[:f]
        rows.append((float(before.sum()), int((before >= 0.05).sum()), float(before.max()) if f else 0.0, pid, f + (len(p_ids) - 1), s))
    hz = sorted(r[0] for r in rows)
    print(f'\n{md.split("/")[-1]}: 128 pass-1 trajectories, up to the base loop start (+300)')
    print(f'  spurious-closure hazard before the intended closure: median {hz[64]:.3f}, p75 {hz[96]:.3f}, p90 {hz[115]:.3f}, max {hz[-1]:.3f} | '
          f'traces with hazard > 0.1: {sum(1 for r in rows if r[0] > 0.1)} | > 0.3: {sum(1 for r in rows if r[0] > 0.3)} | > 1: {sum(1 for r in rows if r[0] > 1)}')
    print(f'  positions with p >= 0.05 before the intended closure: total {sum(r[1] for r in rows)} | traces with any: {sum(1 for r in rows if r[1])}')
    print('  worst traces: ' + ' '.join(f'{pid}:{h:.2f}' for h, k, mx, pid, f, s in sorted(rows, reverse=True)[:8]))
    lib._cache.pop(md, None); del m; torch.cuda.empty_cache()
