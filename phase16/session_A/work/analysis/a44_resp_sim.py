"""Analyse decision B of the response phase (a43 cache): right after a paragraph break, start the
final \boxed{} line or another paragraph. Then simulate a boost of the \boxed token at exactly those
positions. 'Gold mentioned in the response so far' is the proxy for a correct box.
Usage: python a44_resp_sim.py <tag> [D list]"""
import sys
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

tag = sys.argv[1]
Ds = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4, 5, 6, 8]
out = torch.load(f"{tf.CACHE_DIR}/respdec_{tag}.pt", weights_only=False)
tok = tf.tokenizer()
fin = [v for v in out.values() if v["fin"]]; stuck = [v for v in out.values() if not v["fin"]]
print("responses:", len(out), "finished", len(fin), "stuck", len(stuck), "| correct among finished:", sum(v["ok"] for v in fin))
for name, G in [("finished", fin), ("stuck", stuck)]:
    PB = np.array([r["p_box"] for v in G for r in v["B"]]); GL = np.array([r["gold"] for v in G for r in v["B"]])
    ch = np.array([r["next_id"] == 59 for v in G for r in v["B"]])
    gap = np.array([r["lbox"] - r["lmax"] for v in G for r in v["B"]])
    print(f"{name:8s}: paragraph breaks {len(PB)}; p(start \\boxed): mean {PB.mean():.3f}, frac>0.5 {float((PB > 0.5).mean()):.3f}, frac>0.05 {float((PB > 0.05).mean()):.3f}; "
          f"actually chosen {ch.mean():.3f}; raw logit gap to top: deciles {np.quantile(gap, [0.1, 0.5, 0.9]).round(1)}; share after gold mentioned {GL.mean():.3f}")
    gl = [r for v in G for r in v["B"] if r["gold"]]
    g0 = [r for v in G for r in v["B"] if not r["gold"]]
    print(f"          raw logit gap of \\boxed to the top token: after gold mentioned: median {np.median([r['lbox'] - r['lmax'] for r in gl]) if gl else float('nan'):.1f} (n={len(gl)}); "
          f"before: median {np.median([r['lbox'] - r['lmax'] for r in g0]) if g0 else float('nan'):.1f} (n={len(g0)})")
ex = [(v, r) for v in stuck for r in v["B"] if r["gold"]][:16]
print("\nstuck responses, paragraph breaks after the gold answer was mentioned (p(start box), top tokens, context):")
for v, r in ex:
    print(f"  p(box)={r['p_box']:.3f} gap={r['lbox'] - r['lmax']:+6.1f} top={[(repr(tok.decode([a])), p) for a, p in r['top'][:3]]} | ...{r['ctx'][-70:]!r}")
rng = np.random.default_rng(0)
SEEDS = 200
print("\nD = nats added to the \\boxed logit right after a paragraph break of the response (simulation on the recorded responses)")
print(f"{'D':>4s} {'finished% (of 128)':>18s} {'gold@conclude% (of 128)':>24s} {'premature%':>11s} {'stuck-now rescued (gold)':>24s}")
for D in Ds:
    res = []
    for pid, v in out.items():
        c_, g_ = [], []
        for s in range(SEEDS):
            done = False
            for r in v["B"]:
                if D == 0:
                    p = r["p_box"]
                else:
                    p = min(0.999, np.exp((r["lbox"] - r["lmax"] + D) / 0.6))
                if rng.random() < p:
                    c_.append(1); g_.append(bool(r["gold"])); done = True; break
            if not done:
                c_.append(int(v["fin"])); g_.append(bool(v["ok"]))
        res.append((np.mean(c_), np.mean(g_), v["fin"], np.mean(g_)))
    res = np.array(res, dtype=float)
    stuck_res = res[np.array([not v["fin"] for v in out.values()])]
    print(f"{D:4.1f} {100 * res[:, 0].sum() / 128:18.1f} {100 * res[:, 1].sum() / 128:24.1f} {100 * (res[:, 0] - res[:, 1]).sum() / 128:11.1f} {100 * stuck_res[:, 1].sum() / 128:24.1f}")
