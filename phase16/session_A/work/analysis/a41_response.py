"""The response-phase decision: at line ends of the response (after </think>), what does the model
choose - the paragraph break that leads to the final \\boxed{} line, or a doubt word? Compared
between responses that finished and responses that ran to the cap.
Usage: python a41_response.py <pass no> [max generations]"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
maxg = int(sys.argv[2]) if len(sys.argv) > 2 else 80
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load()
E = m.get_input_embeddings().weight
V = E.shape[0]
TE, IMEND = tf.T_THINK_END, tf.T_IM_END
T = {s: tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "Wait", "However", "But", "Hmm", "Therefore", "Thus", "So", "The", "Actually", "Maybe", "Let", "   ", "We", "\\"]}
NLEND = {T["\n"], 271}
SENT = (".", "$", "}", "]", ")")


def read_at(hn, pids, ids, gids, j):
    z = hn[len(pids) + j].float() @ E.float().T              # prediction made at the position of gids[j]
    seen = torch.zeros(V, dtype=torch.bool, device=z.device); seen[ids[: len(pids) + j + 1]] = True
    padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
    v, i = padj.topk(6)
    return dict(j=j, p_nl=float(padj[T["\n"]] + padj[271]), p_wait=float(padj[T["Wait"]]), p_how=float(padj[T["However"]]), p_but=float(padj[T["But"]]),
                p_ind=float(padj[T["   "]]), p_box=float(padj[T["\\"]]), p_end=float(padj[IMEND]),
                top=[(repr(tok.decode([int(a)])), round(float(b), 3)) for a, b in zip(i[:5], v[:5])],
                nxt=tok.decode(gids[j + 1: j + 5]), ctx="".join(tok.decode([x]) for x in gids[max(0, j - 40): j + 1]))


res = []
for r in rows:
    if not r["closed"]:
        continue
    gids = tok.encode(r["text"])
    pids = tf.prompt_ids(pr[r["id"]]["q"])
    te = gids.index(TE)
    start = te + 1
    end = min(len(gids), start + 5000)
    ids = torch.tensor(pids + gids[:end], device=m.device)
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"]
    pts = []
    for j in range(start + 1, end - 2):
        if gids[j] in NLEND:
            s = "".join(tok.decode([x]) for x in gids[max(start, j - 8): j]).rstrip()
            if s.endswith(SENT):
                pts.append(read_at(hn, pids, ids, gids, j))
    # the decision that led to the final \boxed line (finished responses): last line end before the first '\boxed'
    final = None
    if r["finish"] == "stop":
        bx = [i for i in range(start, end - 1) if gids[i] == T["\\"] and tok.decode(gids[i: i + 2]) == "\\boxed"]
        if bx:
            le = [p for p in pts if p["j"] < bx[0]]
            if le:
                final = le[-1]
    res.append(dict(id=r["id"], fin=r["finish"] == "stop", ok=bool(r["ok_boxed"]), pts=pts, final=final, resp_len=len(gids) - start))
    if len(res) >= maxg:
        break
json.dump(res, open(f"{tf.CACHE_DIR}/response_pts_pass{n}.json", "w"))
for grp, name in [(True, "finished responses"), (False, "responses that ran to the cap")]:
    G = [x for x in res if x["fin"] == grp]
    allp = [p for x in G for p in x["pts"]]
    if not allp:
        continue
    print(f"\n=== {name}: {len(G)} responses, {len(allp)} line ends after a sentence ===")
    for key, lab in [("p_wait", "'Wait'"), ("p_how", "'However'"), ("p_but", "'But'"), ("p_nl", "paragraph break"), ("p_ind", "indentation"), ("p_box", "'\\' (start of \\boxed)"), ("p_end", "'<|im_end|>'")]:
        v = np.array([p[key] for p in allp])
        print(f"  p({lab:22s}): mean {v.mean():.3f}  frac>0.5 {float((v > 0.5).mean()):.3f}  frac>0.05 {float((v > 0.05).mean()):.3f}")
    took = np.mean([p["nxt"].lstrip().startswith("Wait") for p in allp])
    print(f"  fraction of these line ends where the model actually wrote 'Wait': {took:.3f}")
    if grp:
        print("  the line end right before the final \\boxed line (what the top candidates were):")
        for x in G:
            if x["final"]:
                f = x["final"]
                print(f"    {x['id']} ok={x['ok']} top={f['top'][:4]} p(nl)={f['p_nl']:.2f} | ...{f['ctx'][-60:]!r}")
    else:
        print("  stuck responses: the first line end at which the model wrote 'Wait':")
        for x in G[:14]:
            w = [p for p in x["pts"] if p["nxt"].lstrip().startswith("Wait")]
            if w:
                f = w[0]
                print(f"    {x['id']} top={f['top'][:4]} p(nl)={f['p_nl']:.3f} | ...{f['ctx'][-75:]!r}")
