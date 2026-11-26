"""Logit scale and teacher-forced accuracy along a generation: sane part vs loop part.

Usage: python exp1.py <pass_no> <problem_id> [model_dir]
"""
import sys, torch
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib

pass_no, pid = int(sys.argv[1]), sys.argv[2]
model_dir = sys.argv[3] if len(sys.argv) > 3 else lib.MODEL_DIR
m, tok = lib.load(model_dir)
prac = {p['id']: p for p in lib.practice()}
g = lib.gens(pass_no)[pid]
p_ids = lib.prompt_ids(tok, prac[pid]['q'])
g_ids = lib.ids_of(tok, g['text'])
ids = p_ids + g_ids
T = len(ids)
print('prompt tokens', len(p_ids), 'gen tokens', len(g_ids), 'recorded', g['n_tokens'])
early = list(range(len(p_ids) - 1, min(T - 1, len(p_ids) + 3000), 40))
late = list(range(max(T - 2500, len(p_ids)), T - 1, 20))
pos = sorted(set(early + late))
r = lib.run(m, ids, positions=pos)
L = r['logits']  # [P,V]
pos_t = torch.tensor(pos)
nxt = torch.tensor(ids)[pos_t + 1].cuda()
top2 = torch.topk(L, 2, dim=-1)
pred = top2.indices[:, 0]
gap = (top2.values[:, 0] - top2.values[:, 1])
maxlog = top2.values[:, 0]
rank_of_next = (L > L.gather(1, nxt[:, None])).sum(1)
for name, sel in [('early', torch.tensor([q in set(early) for q in pos])), ('late', torch.tensor([q in set(late) for q in pos]))]:
    print(f'{name}: n={int(sel.sum())} max-logit mean {maxlog[sel].mean():.2f} (min {maxlog[sel].min():.2f} max {maxlog[sel].max():.2f}); '
          f'top1-top2 gap mean {gap[sel].mean():.2f}; argmax==next {float((pred[sel]==nxt[sel]).float().mean()):.3f}; '
          f'next in top5 {float((rank_of_next[sel]<5).float().mean()):.3f}; logit(next) mean {L.gather(1,nxt[sel,None])[:,0].mean():.2f}')
print('logit std over vocab (mean over positions):', float(L.std(-1).mean()))
THINK = tok.convert_tokens_to_ids('</think>')
print('</think> logit: early mean %.2f, late mean %.2f' % (float(L[torch.tensor([q in set(early) for q in pos]), THINK].mean()), float(L[torch.tensor([q in set(late) for q in pos]), THINK].mean())))
print('sample late positions:')
for j in range(len(pos) - 8, len(pos)):
    print(' pos', pos[j], 'ctx', repr(lib.text_of(tok, ids[pos[j]-12:pos[j]+1])), '-> next', repr(lib.text_of(tok, [ids[pos[j]+1]])) if pos[j] + 1 < T else '-',
          'top:', lib.top_tokens(L[j], tok, 5))
print('sample early positions:')
for j in range(10, 16):
    print(' pos', pos[j], 'ctx', repr(lib.text_of(tok, ids[pos[j]-12:pos[j]+1])), '-> next', repr(lib.text_of(tok, [ids[pos[j]+1]])),
          'top:', lib.top_tokens(L[j], tok, 5))
