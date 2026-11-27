"""Is the response's inability to conclude caused by the long, doubt-laden thinking that precedes it?
For stuck responses of a pass: take the response text up to the first paragraph break that follows
the gold answer, and prepend the thinking truncated to N tokens. Read the raw logit gap of the
final-box token at that paragraph break, as a function of N. Same for finished responses at the
paragraph break that preceded their real final \boxed line.
Usage: python a46_contam.py <pass no> <tag>"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
E = m.get_input_embeddings().weight.float().detach()
TE, BS = tf.T_THINK_END, 59
dec = torch.load(f"{tf.CACHE_DIR}/respdec_{tag}.pt", weights_only=False)
NS = [200, 500, 1000, 2000, 4000, 10 ** 6]
res = {"stuck": {N: [] for N in NS}, "finished": {N: [] for N in NS}}
for pid, v in dec.items():
    r = rows[pid]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE)
    if v["fin"]:
        pts = [x for x in v["B"] if x["next_id"] == BS]
        if not pts:
            continue
        j = pts[-1]["j"]
    else:
        pts = [x for x in v["B"] if x["gold"]]
        if not pts:
            continue
        j = pts[0]["j"]
    grp = "finished" if v["fin"] else "stuck"
    for N in NS:
        cut = min(N, te)
        think = gids[:cut]
        if cut < te and tok.decode([think[-1]]) not in ("\n", "\n\n"):
            think = think + [198]
        ids = torch.tensor(pids + think + [TE] + gids[te + 1: j + 1], device=m.device)
        with torch.no_grad():
            hn = tf.run(m, ids, keep="none")["hn"][-1].float()
            z = hn @ E.T
        res[grp][N].append(float(z[BS] - z.max()))
for grp in ("finished", "stuck"):
    print(f"=== {grp} responses (n={len(res[grp][NS[0]])}): raw logit gap of the final-box token at the paragraph break, by thinking length N ===")
    for N in NS:
        v = np.array(res[grp][N])
        print(f"   N<={N if N < 10**6 else 'full':>6}: median {np.median(v):+6.1f}  mean {v.mean():+6.1f}  deciles {np.quantile(v, [0.1, 0.9]).round(1)}  frac(gap>-2)={float((v > -2).mean()):.2f}")
