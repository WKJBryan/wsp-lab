"""Is 'the answer is X' (a line that states the result) linearly readable from the hidden
activations of the last MLPs at the line end? And would such a readout stay silent on the prompt
and on the response after </think>? (a24 cache)
Usage: python a25_mlp_probe.py <tag>"""
import sys, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

tag = sys.argv[1]
store = torch.load(f"{tf.CACHE_DIR}/mlpfeat_{tag}.pt", weights_only=False)
ANS = re.compile(r"(answer is|answer would be|answer:|answer should be|answers are|final answer)", re.I)
HYP = re.compile(r"(^|\n)\s*(if |maybe|suppose|would |could |what if|in case|assuming|otherwise|but if|unless)", re.I)
PROC = re.compile(r"(wait|let's|let me|check|verify|re-?read|next|step|need to|should i|is it possible|maybe|what if|\?)", re.I)
byid = {g["id"]: g for g in store}
ids, kind, lab, idx = [], [], [], []
for g in store:
    for k in range(g["n_line"]):
        c = g["line_ctx"][k]
        has = bool(ANS.search(c[-70:]))
        l = "ans" if (has and not HYP.search(c)) else ("ans_hyp" if has else ("proc" if PROC.search(c) else "other"))
        ids.append(g["id"]); kind.append("line"); lab.append(l); idx.append(k)
    for k in range(g["n_resp"]):
        ids.append(g["id"]); kind.append("resp"); lab.append("resp"); idx.append(k)
    for k in range(g["n_prompt_pts"]):
        ids.append(g["id"]); kind.append("prompt"); lab.append("prompt"); idx.append(k)
ids = np.array(ids); kind = np.array(kind); lab = np.array(lab); idx = np.array(idx)
rng = np.random.default_rng(0)


def A(layer, sel):
    """MLP hidden activations (input of down_proj) of layer `layer` for the selected rows."""
    pos = np.where(sel)[0]
    res = np.zeros((len(pos), 3584), np.float32)
    for n, p in enumerate(pos):
        g = byid[ids[p]]
        off = g["n_prompt_pts"] if kind[p] == "line" else (g["n_prompt_pts"] + g["n_line"] if kind[p] == "resp" else 0)
        res[n] = g["act"][layer][off + idx[p]]
    return res


def resid(sel):
    pos = np.where(sel)[0]
    res = np.zeros((len(pos), 1024), np.float32)
    for n, p in enumerate(pos):
        g = byid[ids[p]]
        off = g["n_prompt_pts"] if kind[p] == "line" else (g["n_prompt_pts"] + g["n_line"] if kind[p] == "resp" else 0)
        res[n] = g["resid"][off + idx[p]]
    return res


print("rows by kind:", {k: int((kind == k).sum()) for k in ["prompt", "line", "resp"]},
      "| line labels:", {l: int(((lab == l) & (kind == "line")).sum()) for l in ["ans", "ans_hyp", "proc", "other"]})


def cv(X, y, groups, k=5):
    ug = np.array(sorted(set(groups))); perm = rng.permutation(len(ug)); folds = np.array_split(perm, k)
    accs = []
    for f in folds:
        tg = set(ug[f])
        tr = np.array([i for i in range(len(y)) if groups[i] not in tg]); te = np.array([i for i in range(len(y)) if groups[i] in tg])
        w = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        b = 0.5 * (w @ X[tr][y[tr] == 1].mean(0) + w @ X[tr][y[tr] == 0].mean(0))
        pred = X[te] @ w > b; yt = y[te]
        accs.append(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))
    return float(np.mean(accs)), w, b


print("\n=== difference-in-means readout: 'answer stated' lines vs 'process' lines, 5-fold CV by generation ===")
sel = (kind == "line") & ((lab == "ans") | (lab == "proc"))
y = (lab[sel] == "ans").astype(int); gp = ids[sel]
print("n ans:", int(y.sum()), "n proc:", int((1 - y).sum()))
W = {}
for layer in [20, 21, 22, 23]:
    acc, w, b = cv(A(layer, sel), y, gp)
    W[layer] = (w, b)
    print(f"  MLP L{layer} hidden activations: balanced accuracy {acc:.3f}")
print("  final residual stream      : balanced accuracy %.3f" % cv(resid(sel), y, gp)[0])
for name, s in [("answer stated", (kind == "line") & (lab == "ans")), ("hypothetical answer", (kind == "line") & (lab == "ans_hyp")),
                ("process", (kind == "line") & (lab == "proc")), ("other line ends", (kind == "line") & (lab == "other"))]:
    q = np.quantile(np.linalg.norm(A(23, s), axis=1), [0.1, 0.5, 0.9])
    print(f"  |activation| norm at {name:18s}: deciles {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}")

print("\n=== readout score w.a (trained on all 'ans' vs 'proc' lines) on every kind of position, in units of the class gap ===")
w = A(23, sel)[y == 1].mean(0) - A(23, sel)[y == 0].mean(0)
m1, m0 = A(23, sel)[y == 1].mean(0) @ w, A(23, sel)[y == 0].mean(0) @ w
gap = m1 - m0
for name, s in [("line: answer stated", (kind == "line") & (lab == "ans")), ("line: hypothetical answer", (kind == "line") & (lab == "ans_hyp")),
                ("line: process", (kind == "line") & (lab == "proc")), ("line: other", (kind == "line") & (lab == "other")),
                ("prompt tokens", kind == "prompt"), ("response after </think>", kind == "resp")]:
    sc = (A(23, s) @ w - m0) / gap
    q = np.quantile(sc, [0.1, 0.5, 0.9])
    print(f"  {name:28s} n={int(s.sum()):5d}  score deciles 10/50/90 = {q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f}   frac(score>0.5)={float((sc > 0.5).mean()):.3f}  frac(score>1.5)={float((sc > 1.5).mean()):.3f}  max={sc.max():+.2f}")
