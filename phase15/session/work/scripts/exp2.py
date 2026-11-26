"""Per-layer / per-head direct logit attribution at chosen positions of a generation.

Usage: python exp2.py <pass_no> <problem_id> [model_dir] [--early N] [--late N] [--heads]
"""
import sys, torch, argparse
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib

ap = argparse.ArgumentParser()
ap.add_argument('pass_no', type=int); ap.add_argument('pid')
ap.add_argument('model_dir', nargs='?', default=lib.MODEL_DIR)
ap.add_argument('--early', type=int, default=2)
ap.add_argument('--late', type=int, default=4)
ap.add_argument('--heads', action='store_true')
ap.add_argument('--maxlen', type=int, default=0, help='truncate the generation to this many tokens')
ap.add_argument('--tok', default='', help='comma-separated extra tokens to track')
ap.add_argument('--offset', type=int, default=0, help='offset of the late window from the end')
a = ap.parse_args()

m, tok = lib.load(a.model_dir)
prac = {p['id']: p for p in lib.practice()}
g = lib.gens(a.pass_no)[a.pid]
p_ids = lib.prompt_ids(tok, prac[a.pid]['q'])
g_ids = lib.ids_of(tok, g['text'])
if a.maxlen:
    g_ids = g_ids[:a.maxlen]
ids = p_ids + g_ids
T = len(ids)
early = [len(p_ids) + 200 * i + 77 for i in range(a.early)]
late = [T - 1 - a.offset - 40 * i for i in range(a.late)]
pos = sorted(set(early + late))
extra = [tok.convert_tokens_to_ids(t) for t in a.tok.split(',') if t]
THINK = tok.convert_tokens_to_ids('</think>')
r = lib.run(m, ids, positions=pos, sublayers=True)
L = r['logits']
H = r['hidden']
final_h = H[-1][pos].float().cpu()
top2 = torch.topk(L, 2, dim=-1)
print(f'{a.pid}: T={T} early={early} late={late}; </think> id {THINK}')
layers = list(range(24))
A = torch.stack([r['attn'][i] for i in layers])   # [24,P,1024]
M = torch.stack([r['mlp'][i] for i in layers])
dA = lib.dla(m, final_h, [A[i] for i in layers]).cpu()  # [24,P,V]
dM = lib.dla(m, final_h, [M[i] for i in layers]).cpu()
for j, p in enumerate(pos):
    t1, t2 = int(top2.indices[j, 0]), int(top2.indices[j, 1])
    track = [t1, t2] + extra
    ctx = lib.text_of(tok, ids[max(0, p - 10):p + 1])
    print(f'\n== pos {p} {"EARLY" if p in early else "LATE"} ctx={ctx!r}')
    print('   next:', repr(lib.text_of(tok, [ids[p + 1]])) if p + 1 < T else '-', 'top:', lib.top_tokens(L[j], tok, 6))
    print(f'   final logits: ' + ' '.join(f'{lib.text_of(tok,[t])!r}={float(L[j,t]):.2f}' for t in track) + f' </think>={float(L[j,THINK]):.2f}')
    ca = dA[:, j, :][:, track].T   # [n_track, 24]
    cm = dM[:, j, :][:, track].T
    emb = lib.dla(m, final_h[j:j+1], [H[0][p:p+1].cpu()])[0, 0].cpu()
    for k, t in enumerate(track):
        name = lib.text_of(tok, [t])
        line = f'   [{name!r}] emb {float(emb[t]):+.2f} | attn by layer: ' + ' '.join(f'{i}:{float(v):+.1f}' for i, v in enumerate(ca[k]) if abs(float(v)) > 0.8)
        line += ' | mlp by layer: ' + ' '.join(f'{i}:{float(v):+.1f}' for i, v in enumerate(cm[k]) if abs(float(v)) > 0.8)
        print(line)
    th = torch.full((len(track),), THINK)
    cat = dA[:, j, :][:, THINK]
    cmt = dM[:, j, :][:, THINK]
    print(f'   [</think>] attn by layer: ' + ' '.join(f'{i}:{float(v):+.1f}' for i, v in enumerate(cat) if abs(float(v)) > 0.8)
          + ' | mlp by layer: ' + ' '.join(f'{i}:{float(v):+.1f}' for i, v in enumerate(cmt) if abs(float(v)) > 0.8))
    # residual norms
    print('   |attn_out| by layer: ' + ' '.join(f'{i}:{float(A[i,j].norm()):.0f}' for i in range(24)))
    print('   |mlp_out|  by layer: ' + ' '.join(f'{i}:{float(M[i,j].norm()):.0f}' for i in range(24)))
    print('   |residual| by layer  : ' + ' '.join(f'{i}:{float(H[i][p].norm()):.0f}' for i in range(25)))
