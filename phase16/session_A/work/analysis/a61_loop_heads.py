"""Complete table: for every attention / DeltaNet head, the effect of removing it at a loop position
(drop of the loop token's penalised logit; loops whose margin it erases) and at ordinary positions
of the same generations. From the a31 cache.
Usage: python a61_loop_heads.py [min loops broken]"""
import sys
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

minb = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = torch.load(f"{tf.CACHE_DIR}/loopspec_S2.pt", weights_only=False)
N = len(out)
rows = []
for l in range(24):
    kind = "attn" if l % 4 == 3 else "gdn"
    nh = len(out[0]["loop"]["res"][(l, "head")])
    for h in range(nh):
        dl = np.array([o["loop"]["res"][(l, "head")][h][0] for o in out])
        margin = np.array([o["loop"]["base"]["la"] - o["loop"]["base"]["lb"] for o in out])
        dn = np.array([np.mean([x["res"][(l, "head")][h][0] for x in o["normal"]]) for o in out])
        dte = np.array([o["loop"]["res"][(l, "head")][h][1] for o in out])
        rows.append(dict(l=l, kind=kind, h=h, dl=dl.mean(), dlm=np.median(dl), dn=dn.mean(), dnm=np.median(dn), nb=int((dl + margin <= 0).sum()),
                         frac_loop_pos=(dl < -3).mean(), frac_norm_pos=(dn < -1).mean(), dte=np.median(dte)))
rows.sort(key=lambda r: -r["nb"])
print(f"{N} loops; median loop margin {np.median([o['loop']['base']['la'] - o['loop']['base']['lb'] for o in out]):.1f} nats")
print(f"{'head':10s} {'loops broken':>12s} {'drop at loop: mean':>18s} {'median':>7s} {'share of loops with drop<-3':>27s} {'drop at ordinary: mean':>22s} {'median':>7s} {'share ordinary <-1':>19s}")
for r in rows:
    if r["nb"] >= minb:
        print(f"L{r['l']:02d} {r['kind']:4s} h{r['h']:2d} {r['nb']:12d} {r['dl']:18.1f} {r['dlm']:7.1f} {100 * r['frac_loop_pos']:26.0f}% {r['dn']:22.2f} {r['dnm']:7.2f} {100 * r['frac_norm_pos']:18.0f}%")
print("\nnumber of heads breaking at least k loops:", {k: sum(1 for r in rows if r["nb"] >= k) for k in [1, 4, 8, 12, 16, 20, 24, 28]})
torch.save(rows, f"{tf.CACHE_DIR}/loopheads_S2.pt")
