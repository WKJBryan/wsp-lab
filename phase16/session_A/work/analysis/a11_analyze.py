"""Analyse the per-position read of a pass (a10 cache): where does the model ever consider </think>,
and how tight are the loops?
Usage: python a11_analyze.py <pass no> [phrase examples]
CPU only."""
import sys, json, re, glob, os
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
n_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
data = {}
for f in sorted(glob.glob(f"{tf.CACHE_DIR}/a10_pass{n}_*.npz")):
    z = np.load(f)
    pid = os.path.basename(f).split("_")[-1][:-4]
    data[pid] = {k: z[k] for k in z.files}

print("=== per generation: productive phase (before the loop) vs loop phase ===")
print(f"{'id':>4s} {'G':>6s} {'cls':>3s} {'ok':>3s} {'loop@':>6s} | pre-loop: {'max pTE':>8s} {'@pos':>6s} {'mean pTE':>8s} {'mean pAct':>9s} | loop: {'max pTE':>8s} {'mean pAct':>9s} {'min pAct':>8s}")
agg = {"cut_pre_max": [], "cut_loop_max": [], "closed_max": [], "cut_pre_mean": [], "cut_loop_mean": []}
for pid, d in data.items():
    r = rows[pid]
    G = len(d["p_think"])
    L = loops[pid].get("loop_start") if (pid in loops and loops[pid].get("period") and r["finish"] == "length") else None
    pt, pa = d["p_think"], d["p_actual_adj"]
    if L:
        pre, lp = pt[:L], pt[L:]
        s = f"{pre.max():8.4f} {int(pre.argmax()):6d} {pre.mean():8.5f} {pa[:L].mean():9.3f} | {lp.max():8.4f} {lp.mean():9.3f} {np.quantile(lp, 0.001):8.4f}"
        agg["cut_pre_max"].append(pre.max()); agg["cut_loop_max"].append(lp.max())
        agg["cut_pre_mean"].append(pre.mean()); agg["cut_loop_mean"].append(lp.mean())
        at = int(pre.argmax())
    else:
        s = f"{pt.max():8.4f} {int(pt.argmax()):6d} {pt.mean():8.5f} {pa.mean():9.3f} |"
        if r["closed"]:
            agg["closed_max"].append(pt.max())
        else:
            agg["cut_pre_max"].append(pt.max())
        at = int(pt.argmax())
        L = G
    print(f"{pid:>4s} {G:6d} {int(r['closed']):3d} {int(r['ok_boxed']):3d} {str(L):>6s} | {s}")
print()
for k, v in agg.items():
    v = np.array(v)
    if len(v):
        print(f"{k:16s} n={len(v):3d} median={np.median(v):.5f} q90={np.quantile(v, 0.9):.5f} max={v.max():.5f}  n(>0.5)={int((v > 0.5).sum())} n(>0.1)={int((v > 0.1).sum())} n(>0.01)={int((v > 0.01).sum())}")

print("\n=== what the model was about to write at the positions with the highest p(</think>) (pre-loop) ===")
shown = 0
for pid, d in sorted(data.items(), key=lambda kv: -kv[1]["p_think"].max()):
    r = rows[pid]
    if not r["closed"] and shown < 14:
        shown += 1
        L = loops[pid].get("loop_start", len(d["p_think"]))
        at = int(d["p_think"][:L].argmax())
        ids = tok.encode(r["text"])
        ctx = tok.decode(ids[max(0, at - 45): at])
        nxt = tok.decode(ids[at: at + 14])
        print(f"{pid:>4s} pTE={d['p_think'][at]:.4f} (raw logp {d['l_think'][at]:+.1f}) pAct={d['p_actual_adj'][at]:.3f} | ...{ctx!r} >> {nxt!r}")

print("\n=== phrase-conditional p(</think>) over all generations (positions whose preceding text ends with the phrase) ===")
PHRASES = ["This seems correct.", "Ready.", "Okay.", "Final Answer:", "The final answer is", "the final answer is", "So the answer is", "I'll write out the solution now.",
           "I will write the solution.", "Final check", "Let's check", "Wait,", "But wait", "Hmm,", "double-check", "Therefore,", "Thus,", "\\]", "$.", "**"]
res = {p: [] for p in PHRASES}
res_next = {p: [] for p in PHRASES}
for pid, d in data.items():
    r = rows[pid]
    ids = tok.encode(r["text"])
    L = loops[pid].get("loop_start", len(ids)) if pid in loops else len(ids)
    text = r["text"]
    # char offset of each token: recompute by decoding prefixes is slow; use cumulative lengths of decoded tokens
    pieces = [tok.decode([i]) for i in ids]
    ends = np.cumsum([len(p) for p in pieces])
    joined = "".join(pieces)
    for p in PHRASES:
        for m in re.finditer(re.escape(p), joined[: ends[L - 1] if L > 0 else 0]):
            e = m.end()
            i = int(np.searchsorted(ends, e))      # token index whose end == e
            if i < L and ends[i] == e and i + 1 < len(ids):
                res[p].append(d["p_think"][i])
                res_next[p].append(d["p_actual_adj"][i])
for p in PHRASES:
    v = np.array(res[p])
    if len(v):
        print(f"{p!r:34s} n={len(v):5d}  pTE mean={v.mean():.5f} median={np.median(v):.5f} max={v.max():.4f}  n>0.1={int((v > 0.1).sum())}  p(next actual)={np.mean(res_next[p]):.3f}")
