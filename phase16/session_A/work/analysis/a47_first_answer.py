"""The decisive moment of the response: the end of the first sentence that states the answer.
(a) probability of the paragraph break vs the line break there; (b) if a paragraph break is forced
there, with what probability does the model start the final \boxed line? Sampled over many sentence
ends of all closed responses of a pass (to gauge the risk of premature boxes).
Usage: python a47_first_answer.py <pass no> <tag> [points per response]"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
rows = {r["id"]: r for r in tf.gen_rows(n)}
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(tf.BASE_DIR)
E = m.get_input_embeddings().weight
V = E.shape[0]
TE, NL, NL2, BS = tf.T_THINK_END, 198, 271, 59
dec = torch.load(f"{tf.CACHE_DIR}/respdec_{tag}.pt", weights_only=False)
ANS = re.compile(r"(answer|therefore|thus|hence|so the|so, the|is \$?-?[\d\.]+\$?\.?$)", re.I)


def read(ids, j):
    """sampling distribution for the token after position j (model position len = j_abs)"""
    with torch.no_grad():
        hn = tf.run(m, ids[: j + 1], keep="none")["hn"][-1].float()
        z = hn @ E.float().T
    seen = torch.zeros(V, dtype=torch.bool, device=z.device); seen[ids[: j + 1]] = True
    padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
    v, i = padj.topk(4)
    return padj, [(repr(tok.decode([int(a)])), round(float(b), 3)) for a, b in zip(i, v)]


res = []
for pid, v in dec.items():
    r = rows[pid]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE); start = te + 1
    A = v["A"]
    if not A:
        continue
    first_ans = next((x for x in A if x["gold"]), None) or next((x for x in A if ANS.search(x["ctx"])), None)
    pts = []
    if first_ans:
        pts.append(("first_answer", first_ans))
    if v["fin"]:
        last = [x for x in A if x["next_id"] == NL2][-1:]
        pts += [("final_break", x) for x in last]
    step = max(1, len(A) // K)
    pts += [("sample", x) for x in A[::step][:K]]
    for kind, x in pts:
        j = len(pids) + x["j"]
        ids = torch.tensor(pids + gids[: start] + gids[start: x["j"] + 1], device=m.device)
        padj, top = read(ids, j)
        padj2, top2 = read(torch.cat([ids, torch.tensor([NL2], device=m.device)]), j + 1)
        res.append(dict(id=pid, kind=kind, fin=v["fin"], ok=v["ok"], j=x["j"], gold=x["gold"], p_nl2=float(padj[NL2]), p_nl=float(padj[NL]), p_box_forced=float(padj2[BS]),
                        top=top[:3], top_after_forced_break=top2[:3], ctx=x["ctx"][-90:]))
    print(pid, kind, "done", flush=True)
torch.save(res, f"{tf.CACHE_DIR}/firstans_{tag}.pt")
for name, sel in [("stuck responses, first answer statement", lambda x: not x["fin"] and x["kind"] == "first_answer"),
                  ("finished responses, the break before the final box", lambda x: x["fin"] and x["kind"] == "final_break"),
                  ("finished responses, sampled sentence ends", lambda x: x["fin"] and x["kind"] == "sample"),
                  ("stuck responses, sampled sentence ends", lambda x: not x["fin"] and x["kind"] == "sample")]:
    G = [x for x in res if sel(x)]
    if not G:
        continue
    a = np.array([x["p_nl2"] for x in G]); b = np.array([x["p_box_forced"] for x in G]); g = np.array([x["gold"] for x in G])
    print(f"\n=== {name}: n={len(G)} (gold mentioned at the point: {g.mean():.2f}) ===")
    print(f"  p(paragraph break) at the sentence end: mean {a.mean():.3f}, median {np.median(a):.3f}, frac>0.5 {float((a > 0.5).mean()):.2f}")
    print(f"  if the paragraph break is forced: p(start final box): mean {b.mean():.3f}, median {np.median(b):.3f}, frac>0.5 {float((b > 0.5).mean()):.2f}; "
          f"after gold mentioned {np.mean(b[g]) if g.any() else float('nan'):.3f} / before {np.mean(b[~g]) if (~g).any() else float('nan'):.3f}")
for x in [y for y in res if not y["fin"] and y["kind"] == "first_answer"][:14]:
    print(f"  {x['id']} p(nl2)={x['p_nl2']:.3f} p(box | forced break)={x['p_box_forced']:.3f} after-forced top={x['top_after_forced'][:2]} | ...{x['ctx'][-80:]!r}")
