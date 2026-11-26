"""False-alarm analysis of the line-start repeat detector over ALL positions, and a line-start
indicator as a possible second factor.

Inputs: out/linestart_1.pt (line-start residuals, all traces) and out/allpos_1.pt (all positions,
even-id traces). Direction d per layer trained on the TRAIN half of linestart traces (odd ids,
since sorted ids p0,p1,... give odd ids odd indexes); evaluated here on allpos traces with EVEN
index (p1 is not in allpos; allpos holds p0,p2,p4,... at indexes 0,1,2,...; even index = p0,p4,p8,
which are linestart-train traces; odd index = p2,p6,p10,... = held out).

Usage: python allpos_an.py
"""
import sys, torch, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
LS = torch.load('/home/user/wsp14/work/out/linestart_1.pt')
AP = torch.load('/home/user/wsp14/work/out/allpos_1.pt', weights_only=False)
layers = AP['layers']
keys = sorted(LS, key=lambda s: int(s[1:]))
kidx = {k: i for i, k in enumerate(keys)}
cat = lambda f: torch.cat([f(LS[k]) for k in keys])
regions = cat(lambda d: torch.tensor([{'early': 0, 'late': 1, 'loop': 2}[r] for r in d['region']]))
trace = torch.cat([torch.full((len(LS[k]['region']),), i) for i, k in enumerate(keys)])
looping = torch.tensor([any(r == 'loop' for r in LS[k]['region']) for k in keys])[trace]
tr = torch.tensor([i % 2 == 1 for i in range(len(keys))])[trace]      # TRAIN = odd ids (p1,p3,...); allpos holds the even ids (p0,p2,...), so all of it is held out
fresh = (regions == 0) | ((regions == 1) & ~looping)
loop = regions == 2
held = sorted(AP['data'], key=lambda s: int(s[1:]))
print(f'linestart train traces (odd ids): {int(tr.sum())} positions; allpos held-out traces (even ids): {len(held)}')


def auc(pos, neg):
    allv = torch.cat([pos, neg]); r = allv.argsort().argsort().float() + 1
    return float((r[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


for li, L in enumerate(layers):
    X = torch.cat([LS[k]['resid'][L] for k in keys])
    d = X[tr & loop].mean(0) - X[tr & fresh].mean(0)
    print(f'\n===== layer {L} =====')
    s_ls = X @ d
    thr_candidates = [float(torch.quantile(s_ls[loop], q)) for q in (0.05, 0.10, 0.25, 0.50)]
    print(f'  loop line-start score quantiles (all traces): q05 {thr_candidates[0]:+.2f} q10 {thr_candidates[1]:+.2f} q25 {thr_candidates[2]:+.2f} q50 {thr_candidates[3]:+.2f}')
    # line-start indicator direction from HALF of the allpos traces (even position in the sorted list), fresh positions
    ls_pos, ls_neg = [], []
    akeys = sorted(AP['data'], key=lambda s: int(s[1:]))
    for ai, k in enumerate(akeys):
        if ai % 2 == 1:
            continue
        D = AP['data'][k]
        T = D['resid'].shape[1]
        end = D['P0'] + D['loop_start'] - 300 if D['loop_start'] else T
        R = D['resid'][li, :end].float()
        ls_pos.append(R[D['ls'][:end]]); ls_neg.append(R[~D['ls'][:end]][::7])
    ld = torch.cat(ls_pos).mean(0) - torch.cat(ls_neg).mean(0)
    # evaluate on held-out traces
    rows = []
    all_fresh_line, all_fresh_non, all_loop_line, all_loop_non = [], [], [], []
    for k in held:
        D = AP['data'][k]
        R = D['resid'][li].float()
        T = R.shape[0]                       # [T, 1024]
        s = R @ d
        l = R @ ld
        lsm = D['ls']
        if D['loop_start']:
            cut = D['P0'] + D['loop_start']
        else:
            cut = T
        fr = torch.zeros(T, dtype=torch.bool); fr[:cut] = True
        lo = torch.zeros(T, dtype=torch.bool); lo[cut:] = True
        rows.append((k, T, cut, float(s[fr].max()), int(fr.sum()), float(s[fr & lsm].max()) if (fr & lsm).any() else float('nan'),
                     float(s[lo].max()) if lo.any() else float('nan'), float(s[lo].mean()) if lo.any() else float('nan')))
        all_fresh_line.append(s[fr & lsm]); all_fresh_non.append(s[fr & ~lsm])
        all_loop_line.append(s[lo & lsm]); all_loop_non.append(s[lo & ~lsm])
    afn = torch.cat(all_fresh_non); afl = torch.cat(all_fresh_line)
    all_l = torch.cat(all_loop_line); aln = torch.cat(all_loop_non)
    print(f'  held-out FRESH, all positions ({afl.numel() + afn.numel()}): line starts max {float(afl.max()):+.2f} q99.9 {float(torch.quantile(afl, 0.999)):+.2f} | '
          f'non-line-start max {float(afn.max()):+.2f} q99.99 {float(torch.quantile(afn, 0.9999)):+.2f}')
    print(f'  per trace fresh max, line-start / any position: ' + ' '.join(f'{r[0]}:{r[5]:+.2f}/{r[3]:+.2f}' for r in rows))
    fmax = max(r[3] for r in rows)
    for q, thr in zip((0.05, 0.10, 0.25, 0.50), thr_candidates):
        print(f'  thr {thr:+.2f} (loop line-start q{int(q*100):02d}): fresh positions above: line-start {int((afl > thr).sum())}/{afl.numel()}  any {int((afn > thr).sum())}/{afn.numel()}  | '
              f'loop positions above: line-start {float((all_l > thr).float().mean()):.2f}  non-line-start {float((aln > thr).float().mean()):.2f}')
    # line-start indicator quality on the OTHER half of the allpos traces
    lsl, lsn = [], []
    for ai, k in enumerate(akeys):
        if ai % 2 == 0:
            continue
        D = AP['data'][k]
        R = D['resid'][li].float(); T = R.shape[0]
        l = R @ ld; lsm = D['ls']
        cut = D['P0'] + D['loop_start'] if D['loop_start'] else T
        lsl.append(l[lsm & (torch.arange(T) < cut)]); lsn.append(l[(~lsm) & (torch.arange(T) < cut)][::5])
    lsl, lsn = torch.cat(lsl), torch.cat(lsn)
    print(f'  line-start indicator at layer {L}: AUC {auc(lsl, lsn):.4f}; line-start scores q01 {float(torch.quantile(lsl, 0.01)):+.2f} q05 {float(torch.quantile(lsl, 0.05)):+.2f} med {float(lsl.median()):+.2f} | '
          f'others q99 {float(torch.quantile(lsn, 0.99)):+.2f} q99.9 {float(torch.quantile(lsn, 0.999)):+.2f} max {float(lsn.max()):+.2f}')
