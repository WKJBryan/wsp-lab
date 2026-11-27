"""How often does the model announce that it is about to write the response, and how close to
</think> is it right after such lines? Uses the a10 per-position cache.
Usage: python a12_intentions.py <pass no>"""
import sys, json, re, glob, os
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
rows = {r["id"]: r for r in tf.gen_rows(n)}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
INTENT = re.compile(r"(write (out |the |down )?(the )?(response|solution|answer|final)|I will (write|provide|present|formulate|generate|give|start)|I'll (write|provide|present|formulate|generate|give|start|just)"
                    r"|final output|generate the response|thinking process is complete|done thinking|time to (write|respond|answer)|ready to (respond|answer|generate)|"
                    r"final answer|the answer is|so the answer|answer:|therefore, the answer|conclusion:|in conclusion|to summarize|in summary|wrap up|sum up|finalize|finalise)", re.I)
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
tot = {"gen": 0, "intent_pts": 0, "intent_ge01": 0, "intent_ge003": 0, "gen_with": 0, "gen_ge01": 0, "first_intent_t": [], "first_ge01_t": []}
per_gen = []
examples = []
for f in sorted(glob.glob(f"{tf.CACHE_DIR}/a10_pass{n}_*.npz")):
    pid = os.path.basename(f).split("_")[-1][:-4]
    r = rows[pid]
    d = np.load(f)
    gids = tok.encode(r["text"])
    G = len(d["p_think"])
    L = loops[pid].get("loop_start", G) if pid in loops else G
    te = gids.index(tf.T_THINK_END) if tf.T_THINK_END in gids else G
    L = min(L, te)
    pieces = [tok.decode([x]) for x in gids]
    ends = np.cumsum([len(p) for p in pieces])
    joined = "".join(pieces)
    pts = []
    for m in INTENT.finditer(joined[: int(ends[L - 1]) if L > 0 else 0]):
        e = m.end()
        # the decision point: the first newline token after the end of this line
        k = int(np.searchsorted(ends, e))
        j = k
        while j < L and gids[j] not in nl_ids and j < k + 60:
            j += 1
        if j < L and gids[j] in nl_ids:
            pts.append(j)
    pts = sorted(set(pts))
    tot["gen"] += 1
    tot["intent_pts"] += len(pts)
    if pts:
        tot["gen_with"] += 1
        tot["first_intent_t"].append(pts[0])
    hi = [j for j in pts if d["p_think"][j] >= 0.01]
    mid = [j for j in pts if d["p_think"][j] >= 0.003]
    tot["intent_ge01"] += len(hi); tot["intent_ge003"] += len(mid)
    if hi:
        tot["gen_ge01"] += 1
        tot["first_ge01_t"].append(hi[0])
    per_gen.append((pid, r["closed"], len(pts), len(mid), len(hi), max([d["p_think"][j] for j in pts], default=0.0), pts[0] if pts else -1))
    if not r["closed"] and len(examples) < 25 and pts:
        j = pts[len(pts) // 2]
        examples.append((pid, j, float(d["p_think"][j]), joined[int(ends[max(0, j - 60)]): int(ends[j + 3]) if j + 3 < len(ends) else len(joined)]))
print(tot)
print("token of the first intent line: quartiles", np.quantile(tot["first_intent_t"], [0.1, 0.25, 0.5, 0.75, 0.9]).round(0) if tot["first_intent_t"] else None)
print("token of the first intent point with p>=0.01: quartiles", np.quantile(tot["first_ge01_t"], [0.1, 0.25, 0.5, 0.75, 0.9]).round(0) if tot["first_ge01_t"] else None)
print("\nper generation (cut-off only): id, #intent points, #p>=0.003, #p>=0.01, max pTE, first intent token")
for pid, closed, a, b, c, mx, t0 in per_gen:
    if not closed:
        print(f"  {pid:>4s} n={a:3d} mid={b:2d} hi={c:2d} maxp={mx:.4f} first@{t0}")
print("\nexamples (middle intent point of cut-off generations):")
for pid, j, p, s in examples:
    print(f"  {pid:>4s} pTE={p:.4f} | ...{s!r}")
