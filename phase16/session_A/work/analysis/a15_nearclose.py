"""Near-closing moments: positions (after a line break) where </think> is within a few nats of the
top prediction. How often do they occur, and what precedes them?
Usage: python a15_nearclose.py <pass no> [threshold] [examples]"""
import sys, json, glob, os
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
thr = float(sys.argv[2]) if len(sys.argv) > 2 else -6.0
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 45
rows = {r["id"]: r for r in tf.gen_rows(n)}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
thr_all = [-3.5, -4.0, -5.0, -6.0, -7.0]
cnt = {t: [] for t in thr_all}
first = {t: [] for t in thr_all}
cut_flag = []
ctx = []
for f in sorted(glob.glob(f"{tf.CACHE_DIR}/a10_pass{n}_*.npz")):
    pid = os.path.basename(f).split("_")[-1][:-4]
    r = rows[pid]
    d = np.load(f)
    gids = tok.encode(r["text"])
    G = len(d["l_think"])
    L = loops[pid].get("loop_start", G) if (pid in loops and loops[pid].get("period")) else G
    te = gids.index(tf.T_THINK_END) if tf.T_THINK_END in gids else G
    L = min(L, te, G)
    lt = np.where(np.arange(G) > 5, d["l_think"], -99.0)[:L]
    ends = np.cumsum([len(tok.decode([x])) for x in gids])
    joined = "".join(tok.decode([x]) for x in gids[: L + 30])
    cut_flag.append(not r["closed"])
    for t in thr_all:
        w = np.where(lt > t)[0]
        cnt[t].append(len(w))
        first[t].append(int(w[0]) if len(w) else -1)
    if not r["closed"]:
        for j in np.where(lt > thr)[0][:3]:
            a = int(ends[max(0, j - 75)]); b = int(ends[j]) if j < len(ends) else len(joined)
            ctx.append((pid, int(j), float(lt[j]), joined[a:b], joined[b: b + 22]))
cut_flag = np.array(cut_flag)
for t in thr_all:
    c = np.array(cnt[t]); fi = np.array(first[t])
    print(f"raw logp(</think>) > {t:+.1f}: generations with >=1: {int((c >= 1).sum())}/128 (cut-off: {int((c[cut_flag] >= 1).sum())}/121); "
          f">=2: {int((c >= 2).sum())}; mean count {c.mean():.1f}; median {np.median(c):.0f}; "
          f"first such position quartiles {np.quantile(fi[fi >= 0], [0.1, 0.5, 0.9]).round(0) if (fi >= 0).any() else None}")
print(f"\ncontexts of near-closing moments in cut-off generations (first three per generation with logp > {thr}):")
for pid, j, v, s, nxt in ctx[:n_ex]:
    print(f"{pid:>4s} t={j:5d} logpTE={v:6.2f} | ...{s!r} >> {nxt!r}")
