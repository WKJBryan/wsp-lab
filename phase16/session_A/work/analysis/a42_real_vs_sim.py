"""Compare the real closings of a pass with (a) the baseline trajectories (identical text up to the
closing point?) and (b) the simulated closing probability of the edit at the real closing point.
Usage: python a42_real_vs_sim.py <pass no> <k>"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); k = float(sys.argv[2])
r1 = {r["id"]: r for r in tf.gen_rows(1)}; rn = {r["id"]: r for r in tf.gen_rows(n)}
tok = tf.tokenizer()
out = torch.load(f"{tf.CACHE_DIR}/editk_p1.pt", weights_only=False)
TE = tf.T_THINK_END
nl_ids = {198, 271}


def p_te(k, tops, cs, first, t):
    ids_ = tops[:, 0].astype(int); z = tops[:, 1].astype(np.float64).copy(); z[ids_ == TE] += (k - 1.0) * cs
    pen = np.array([1.5 if first.get(int(i), 10 ** 9) <= t else 0.0 for i in ids_]); z = (z - pen) / 0.6
    o = np.argsort(-z)[:20]; zk = z[o]; p = np.exp(zk - zk.max()); p /= p.sum(); c = np.cumsum(p); p = p * ((c - p) < 0.95); p /= p.sum()
    return float(p[ids_[o] == TE].sum())


rows = []
for pid, g in out.items():
    r = rn[pid]
    t1 = tok.encode(r1[pid]["text"]); t2 = tok.encode(r["text"])
    te2 = t2.index(TE) if TE in t2 else None
    n = 0
    for a, b in zip(t1, t2):
        if a != b:
            break
        n += 1
    if te2 is None:
        rows.append(dict(id=pid, closed=False, same=n / max(1, len(t1)), loop=g["loop_start"], fin=False))
        continue
    gids = g["gids"]
    first = {}
    for j, x in enumerate(gids):
        first.setdefault(int(x), j)
    t = te2 - 1
    p = p_te(k, g["tops"][t + 1], g["cs"][t + 1], first, t) if t + 1 < len(g["tops"]) else float("nan")
    rows.append(dict(id=pid, closed=True, same=n / te2, identical=n >= te2, loop=g["loop_start"], te2=te2, p=p, fin=r["finish"] == "stop", ok=bool(r["ok_boxed"]),
                     inloop=bool(g["loop_start"] and te2 > g["loop_start"])))
cl = [x for x in rows if x["closed"]]
print(f"pass {n}, k={k}: closed {len(cl)} of {len(rows)}; identical to the baseline text up to the closing point: {sum(x['identical'] for x in cl)} of {len(cl)}; "
      f"median shared share {np.median([x['same'] for x in cl]):.3f}")
print(f"closings inside the baseline loop region: {sum(x['inloop'] for x in cl)}; finished: {sum(x['fin'] for x in cl)}; of the closings before the loop: finished {sum(x['fin'] for x in cl if not x['inloop'])} of {sum(1 for x in cl if not x['inloop'])}")
bl = [x["te2"] for x in cl if not x["inloop"]]; il = [x["te2"] for x in cl if x["inloop"]]
print("closing token: before-loop closings quartiles", np.quantile(bl, [0.1, 0.5, 0.9]).round(0) if bl else None,
      "| in-loop closings quartiles", np.quantile(il, [0.1, 0.5, 0.9]).round(0) if il else None)
ps = [x["p"] for x in cl if not np.isnan(x["p"])]
print(f"simulated p(</think>) at the real closing points (where available): median {np.median(ps):.4f}, quartiles {np.quantile(ps, [0.25, 0.75]).round(4)} (n={len(ps)})")
