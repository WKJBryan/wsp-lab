"""Is there a linear residual-stream signal for 'this context is verbatim-repeating'?

For every pass-1 trace with a verbatim loop:
  REP   = sampled token positions deep inside the loop (from cycle 10 on)
  FRESH = sampled token positions in the first min(loop_start, 4000) generated tokens
For each layer (residual at the input of layer L; 24 = final): a mean-difference direction is
computed on half of the traces and scored on the other half. Reported: AUC on held-out traces,
and, with the threshold set just above the highest FRESH score of the training half, the recall on
held-out REP positions and the false-positive rate on held-out FRESH positions.

Usage: python repfeat.py [--pass 1] [--n 113] [--model_dir DIR] [--save out/repfeat.pt]
"""
import sys, torch, argparse, json, random
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
ap = argparse.ArgumentParser()
ap.add_argument('--pass', dest='pass_no', type=int, default=1)
ap.add_argument('--n', type=int, default=113)
ap.add_argument('--cycles', type=int, default=10)
ap.add_argument('--per', type=int, default=10)
ap.add_argument('--model_dir', default=lib.MODEL_DIR)
ap.add_argument('--save', default='')
ap.add_argument('--load', default='')
ap.add_argument('--freshmax', type=int, default=4000)
a = ap.parse_args()
prac = {p['id']: p for p in lib.practice()}
G = lib.gens(a.pass_no)
loops = json.load(open(f'/home/user/wsp14/work/out/loops_{a.pass_no}.json'))
pids = [pid for pid in sorted(loops, key=lambda s: int(s[1:])) if loops[pid]['loop_start'] and loops[pid]['period'] >= 6][:a.n]
if a.load:
    data = torch.load(a.load)
else:
    m, tok = lib.load(a.model_dir)
    data = {}
    rng = random.Random(0)
    for n, pid in enumerate(pids):
        s, P = loops[pid]['loop_start'], loops[pid]['period']
        p_ids = lib.prompt_ids(tok, prac[pid]['q'])
        ids = p_ids + lib.ids_of(tok, G[pid]['text'])
        T = len(ids)
        lo = len(p_ids) + s + a.cycles * P
        rep = [rng.randrange(lo, T - 2) for _ in range(a.per)]          # lo is already absolute
        hi = min(s, a.freshmax)
        fresh = [len(p_ids) + rng.randrange(100, hi) for _ in range(a.per)]
        r = lib.run_pos(m, ids, rep + fresh)
        data[pid] = dict(resid=r['resid'], nrep=a.per)          # [25, 2*per, 1024]
        if n % 20 == 0:
            print(f'  {n}/{len(pids)} traces done', flush=True)
    if a.save:
        torch.save(data, a.save)
print(f'{len(data)} traces; residual tensors {next(iter(data.values()))["resid"].shape}')


def auc(pos, neg):
    pos, neg = pos.flatten(), neg.flatten()
    allv = torch.cat([pos, neg])
    ranks = allv.argsort().argsort().float() + 1
    return float((ranks[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


keys = sorted(data, key=lambda s: int(s[1:]))
tr, te = keys[0::2], keys[1::2]
R = torch.stack([data[k]['resid'] for k in keys])      # [N, 25, 2*per, 1024]
per = data[keys[0]]['nrep']
is_tr = torch.tensor([k in tr for k in keys])
REP = R[:, :, :per]; FRESH = R[:, :, per:]
print('layer | norm ratio |REP-FRESH mean| / |h| (train) | cos of per-trace difference with the common direction (train, test) | AUC test | recall@FPR0 train-threshold (test REP, test FRESH)')
rows = []
for L in range(25):
    rep_tr, fr_tr = REP[is_tr, L], FRESH[is_tr, L]
    d = rep_tr.reshape(-1, 1024).mean(0) - fr_tr.reshape(-1, 1024).mean(0)
    c_tr = torch.nn.functional.cosine_similarity(rep_tr.mean(1) - fr_tr.mean(1), d[None], dim=1).mean()
    rep_te, fr_te = REP[~is_tr, L], FRESH[~is_tr, L]
    c_te = torch.nn.functional.cosine_similarity(rep_te.mean(1) - fr_te.mean(1), d[None], dim=1).mean()
    s_tr_r, s_tr_f = rep_tr @ d, fr_tr @ d
    s_te_r, s_te_f = rep_te @ d, fr_te @ d
    thr = float(s_tr_f.max())
    hn = fr_tr.reshape(-1, 1024).norm(dim=1).mean()
    rows.append((L, d, thr))
    print(f'  {L:2d}  | {float(rep_tr.norm(dim=-1).mean() / hn):.2f} | {float(d.norm() / hn):.3f} | {float(c_tr):+.2f} {float(c_te):+.2f} | {auc(s_te_r, s_te_f):.3f} | '
          f'{float((s_te_r > thr).float().mean()):.2f}  {float((s_te_f > thr).float().mean()):.3f}   '
          f'(score REP test mean {float(s_te_r.mean()):+.1f} sd {float(s_te_r.std()):.1f}; FRESH test mean {float(s_te_f.mean()):+.1f} sd {float(s_te_f.std()):.1f}; thr {thr:+.1f})')
if a.save:
    torch.save(dict(rows=[(L, d, thr) for L, d, thr in rows]), a.save.replace('.pt', '_dirs.pt'))
