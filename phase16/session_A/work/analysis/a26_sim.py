"""Simulation on the baseline trajectories (teacher-forced data of a10): if the </think> logit at
line ends were raised by D nats (all else equal), when would the model close, and had it already
written the gold answer by then?  'Gold mentioned before closing' is an upper-ish proxy for a
correct final answer. Generations that closed in the baseline keep their real outcome if the
simulation does not close them earlier.
Usage: python a26_sim.py <pass no>"""
import sys, json, glob, os, re
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
TEMP, PEN = 0.6, 1.5
SEEDS = 300


def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s).replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "").strip("$")
    return s


data = {}
for f in sorted(glob.glob(f"{tf.CACHE_DIR}/a10_pass{n}_*.npz")):
    pid = os.path.basename(f).split("_")[-1][:-4]
    d = np.load(f)
    gids = tok.encode(rows[pid]["text"])
    G = len(d["p_think"])
    le = [t for t in range(8, G - 1) if gids[t] in nl_ids]
    g = norm(pr[pid]["gold"])
    pieces = [norm(tok.decode([x])) for x in gids]
    gm, pref, j = {}, "", 0
    for t in le:
        while j <= t:
            pref += pieces[j]; j += 1
        gm[t] = (g != "") and (g in pref)
    te = gids.index(tf.T_THINK_END) if tf.T_THINK_END in gids else None
    data[pid] = dict(le=le, gm=gm, p=d["p_think"], l=d["l_think"], te=te, closed=bool(rows[pid]["closed"]),
                     ok=bool(rows[pid]["ok_boxed"]), lvl=pr[pid]["level"])

print("gold answer mentioned somewhere before the last line end used:", sum(1 for v in data.values() if any(v["gm"][t] for t in v["le"])), "of", len(data))


def p_edit(p0, l0, D):
    """sampling probability of </think> at a line end after adding D nats to its logit
    (approximation: odds against the seen top token, presence penalty included)."""
    x = (l0 + D) / TEMP + PEN
    if x < -60:
        return 0.0
    p = 1.0 / (1.0 + np.exp(-x))
    return min(p, 0.999)


# consistency of the approximation with the exact adjusted probabilities at D = 0
rat = [p_edit(float(v["p"][t + 1]), float(v["l"][t + 1]), 0) / float(v["p"][t + 1]) for v in data.values() for t in v["le"] if 1e-4 < float(v["p"][t + 1]) < 0.5]
if rat:
    print("approximation check (D=0), ratio approx/exact at line ends with 1e-4<p<0.5: median %.2f, deciles %.2f / %.2f (n=%d)" % (np.median(rat), np.quantile(rat, 0.1), np.quantile(rat, 0.9), len(rat)))
print("line ends where </think> is in the top-20:", sum(1 for v in data.values() for t in v["le"] if float(v["p"][t + 1]) > 0), "of", sum(len(v["le"]) for v in data.values()))
rng = np.random.default_rng(0)


print("\nD = nats added to the </think> logit at line ends (the prediction right after a newline)")
print(f"{'D':>4s} {'closed%':>8s} {'gold@close%':>12s} {'premature%':>11s} {'close<600tok%':>14s} {'median close tok':>17s}   gold@close by level (1..5)")
for D in [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]:
    allc, allg, allprem, allearly, byl = [], [], [], [], {1: [], 2: [], 3: [], 4: [], 5: []}
    toks = []
    for pid, v in data.items():
        c_, g_ = [], []
        for s in range(SEEDS):
            done = False
            for t in v["le"]:          # prediction of the token after the newline = index t+1
                if rng.random() < p_edit(float(v["p"][t + 1]), float(v["l"][t + 1]), D):
                    c_.append(t); g_.append(bool(v["gm"][t])); done = True; break
            if not done:
                if v["te"] is not None:
                    c_.append(v["te"]); g_.append(v["ok"])
                else:
                    c_.append(-1); g_.append(False)
        c_ = np.array(c_); g_ = np.array(g_)
        cl = c_ >= 0
        allc.append(cl); allg.append(g_); allprem.append(cl & ~g_); allearly.append(cl & (c_ < 600)); toks.append(c_[cl])
        byl[int(v["lvl"][-1])].append((g_[cl].mean() if cl.any() else 0.0, cl.any()))
    allc = np.concatenate(allc); allg = np.concatenate(allg); allprem = np.concatenate(allprem); allearly = np.concatenate(allearly)
    toks = np.concatenate(toks)
    print(f"{D:4d} {100 * allc.mean():8.1f} {100 * allg.mean():12.1f} {100 * allprem.mean():11.1f} {100 * allearly.mean():14.1f} {np.median(toks):17.0f}   " +
          "  ".join(f"L{k}:{100 * np.mean([x[0] for x in v]):.0f}({100 * np.mean([x[1] for x in v]):.0f}c)" for k, v in byl.items()))
