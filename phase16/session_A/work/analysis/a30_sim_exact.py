"""Exact simulation (evaluation sampling rules) of closing behaviour when the </think>-aligned
output of the line-end-specific closing neurons of the layer-23 MLP is multiplied by k.
Teacher-forced on the baseline trajectories; 'gold mentioned before closing' is the proxy for a
correct final answer. Baseline-closed generations keep their real outcome if not closed earlier.
Usage: python a30_sim_exact.py <tag>"""
import sys, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

tag = sys.argv[1]
out = torch.load(f"{tf.CACHE_DIR}/editk_{tag}.pt", weights_only=False)
tok = tf.tokenizer()
TE = tf.T_THINK_END
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
TEMP, PEN, TOPK, TOPP = 0.6, 1.5, 20, 0.95
SEEDS = 200


def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s).replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "").strip("$")
    return s


def p_te(k, tops, cs, first, t):
    """sampling probability of </think> at the position whose prediction is token index t+1.
    `first`: dict token id -> index of its first occurrence in the generation."""
    ids_ = tops[:, 0].astype(int); z = tops[:, 1].astype(np.float64).copy()
    z[ids_ == TE] += (k - 1.0) * cs
    pen = np.array([PEN if first.get(int(i), 10 ** 9) <= t else 0.0 for i in ids_])
    z = (z - pen) / TEMP
    order = np.argsort(-z)[:TOPK]
    zk = z[order]; p = np.exp(zk - zk.max()); p /= p.sum()
    c = np.cumsum(p); keep = (c - p) < TOPP
    p = p * keep; p /= p.sum()
    return float(p[ids_[order] == TE].sum())


# prepare per generation: line ends, gold-mentioned flags, first-occurrence table
G = {}
for pid, g in out.items():
    gids = g["gids"]
    limit = g["loop_start"] if g["loop_start"] else (g["te"] if g["te"] is not None else len(gids) - 3)
    limit = min(limit, len(gids) - 3)
    le = [t for t in range(3, min(limit, len(gids) - 2)) if gids[t] in nl_ids]
    gold = norm(g["gold"])
    pieces = [norm(tok.decode([int(x)])) for x in gids[: (max(le) + 2 if le else 10)]]
    gm, pref, first, j, recent = {}, "", {}, 0, ""
    # gm: gold mentioned anywhere before the line end; gr: gold mentioned in the last ~160 characters before it
    gr = {}
    for t in le:
        while j <= t:
            pref += pieces[j]; first.setdefault(int(gids[j]), j); recent = (recent + pieces[j])[-160:]; j += 1
        gm[t] = (gold != "") and (gold in pref)
        gr[t] = (gold != "") and (gold in recent)
    G[pid] = dict(g=g, le=le, gm=gm, gr=gr, cs=g["cs"], limit=limit, first=first)
rng = np.random.default_rng(0)
KS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2.0, 2.1, 2.2, 2.3]
print(f"{'k':>5s} {'boost':>6s} {'closed%':>8s} {'gold@close%':>12s} {'recent-gold%':>13s} {'premature%':>11s} {'close<400tok%':>14s} {'median tok':>11s} {'mean tok':>9s}  gold@close by level 1..5")
for k in KS:
    allc, allg, allr, allprem, allearly, toks = [], [], [], [], [], []
    byl = {1: [], 2: [], 3: [], 4: [], 5: []}
    boost = []
    mid = []
    for pid, v in G.items():
        g = v["g"]
        ps = [p_te(k, g["tops"][t + 1], v["cs"][t + 1], v["first"], t) for t in v["le"]]
        boost += [(k - 1) * float(v["cs"][t + 1]) for t in v["le"][:40]]
        # mid-line positions (every 40th token): closing probability must stay ~0
        for t in range(20, v["limit"], 40):
            if g["gids"][t] not in nl_ids:
                mid.append(p_te(k, g["tops"][t + 1], v["cs"][t + 1], v["first"], t))
        c_, g_, r_ = [], [], []
        for s in range(SEEDS):
            done = False
            for t, p in zip(v["le"], ps):
                if rng.random() < p:
                    c_.append(t); g_.append(bool(v["gm"][t])); r_.append(bool(v["gr"][t])); done = True; break
            if not done:
                if g["te"] is not None:
                    c_.append(g["te"]); g_.append(g["ok"]); r_.append(g["ok"])
                else:
                    c_.append(-1); g_.append(False); r_.append(False)
        c_ = np.array(c_); g_ = np.array(g_); r_ = np.array(r_); cl = c_ >= 0
        allc.append(cl); allg.append(g_); allr.append(r_); allprem.append(cl & ~g_); allearly.append(cl & (c_ < 400)); toks.append(c_[cl])
        byl[int(g["level"][-1])].append((g_[cl].mean() if cl.any() else 0.0, cl.any()))
    allc = np.concatenate(allc); allg = np.concatenate(allg); allr = np.concatenate(allr); allprem = np.concatenate(allprem); allearly = np.concatenate(allearly)
    toks = np.concatenate(toks) if toks else np.array([0])
    print(f"{k:5.2f} {np.median(boost):6.2f} {100 * allc.mean():8.1f} {100 * allg.mean():12.1f} {100 * allr.mean():13.1f} {100 * allprem.mean():11.1f} {100 * allearly.mean():14.1f} {np.median(toks):11.0f} {toks.mean():9.0f}  " +
          "  ".join(f"L{L}:{100 * np.mean([x[0] for x in w]):.0f}({100 * np.mean([x[1] for x in w]):.0f}c)" for L, w in byl.items()) +
          f"   | mid-line p(</think>): mean {np.mean(mid):.2e} max {np.max(mid):.3f}")
