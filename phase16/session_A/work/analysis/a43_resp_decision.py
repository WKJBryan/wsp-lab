"""Record the response-phase decisions on real responses (teacher-forced):
 A. at every sentence end of the response (token right before a line break): paragraph break vs line break
 B. right after a paragraph break ("\n\n"): start the final \boxed{} line or another paragraph
Writes work/cache/respdec_<tag>.pt
Usage: python a43_resp_decision.py <pass no> <tag> [max response tokens]"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); tag = sys.argv[2]
maxr = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load()
E = m.get_input_embeddings().weight
V = E.shape[0]
TE, IMEND = tf.T_THINK_END, tf.T_IM_END
NL, NL2, BS = 198, 271, 59       # "\n", "\n\n", "\\" (first token of \boxed)
SENT = (".", "!", "?", "$$", "$.", "}.", ").", "].")


def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s).replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "").strip("$")
    return s


out = {}
for r in rows:
    if not r["closed"]:
        continue
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    te = gids.index(TE); start = te + 1
    end = min(len(gids), start + maxr)
    ids = torch.tensor(pids + gids[: end + 1], device=m.device)
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"]
    pieces = [tok.decode([x]) for x in gids]
    A = [j for j in range(start, end - 1) if gids[j + 1] in (NL, NL2) and pieces[j].rstrip().endswith(SENT)]
    B = [j for j in range(start, end - 1) if gids[j] == NL2 or (gids[j] == NL and gids[j - 1] == NL)]
    A, B = A[:300], B[:300]
    gold = norm(pr[pid]["gold"])
    gold_cum = {}
    resp_norm = ""; k = start
    for j in sorted(set(A) | set(B)):
        while k <= j:
            resp_norm += norm(pieces[k]); k += 1
        gold_cum[j] = bool(gold) and (gold in resp_norm)

    def rec(j):
        z = hn[len(pids) + j].float() @ E.float().T
        lse = torch.logsumexp(z, -1)
        seen = torch.zeros(V, dtype=torch.bool, device=z.device); seen[ids[: len(pids) + j + 1]] = True
        padj = tf.adj_sample_prob(z[None], seen=seen[None])[0]
        v, i = padj.topk(5)
        return dict(j=j, p_nl2=float(padj[NL2]), p_nl=float(padj[NL]), p_box=float(padj[BS]), p_end=float(padj[IMEND]), next_id=int(gids[j + 1]) if j + 1 < len(gids) else -1,
                    l272=float(z[NL2] - lse), l198=float(z[NL] - lse), lbox=float(z[BS] - lse), lmax=float(z.max() - lse),
                    top=[(int(a), round(float(b), 3)) for a, b in zip(i, v)], gold=gold_cum[j], ctx="".join(pieces[max(start, j - 30): j + 1]))
    out[pid] = dict(A=[rec(j) for j in A], B=[rec(j) for j in B], fin=r["finish"] == "stop", ok=bool(r["ok_boxed"]), resp_len=len(gids) - start, te=te)
    print(pid, "A", len(A), "B", len(B), "finished" if r["finish"] == "stop" else "stuck", flush=True)
torch.save(out, f"{tf.CACHE_DIR}/respdec_{tag}.pt")
print("saved", len(out))
