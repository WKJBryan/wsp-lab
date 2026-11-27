"""Is the response's 'doubt mode' linearly readable where it matters (at paragraph breaks of the
response)? Collect layer-23 MLP hidden activations and the layer-23 input residual at paragraph
breaks of every closed response of a pass, labelled stuck / healthy(non-final) / healthy(final),
plus thinking-phase paragraph breaks. Cross-validated difference-in-means probes.
Usage: python a49_doubt_probe.py <model dir> <pass no> <tag>"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

model_dir, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(model_dir)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
TE, NL, NL2, BS = tf.T_THINK_END, 198, 271, 59
X, R, Y, GID, TPOS, GAP = [], [], [], [], [], []
for r in rows:
    if not r["closed"]:
        continue
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE); start = te + 1
    end = min(len(gids), start + 4000)
    ids = torch.tensor(pids + gids[: end + 1], device=m.device)
    act = {}; res_h = []; outs = []
    hooks = [Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0].detach())),
             Lm.layers[23].register_forward_pre_hook(lambda mod, args, kwargs: res_h.append(args[0][0].detach()), with_kwargs=True)]
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"]
    for h in hooks:
        h.remove()
    pb_resp = [j for j in range(start + 1, end - 1) if gids[j] == NL2 or (gids[j] == NL and gids[j - 1] == NL)]
    pb_think = [j for j in range(5, te - 1) if gids[j] == NL2 or (gids[j] == NL and gids[j - 1] == NL)]
    fin = r["finish"] == "stop"
    pts = [(j, "resp") for j in pb_resp[:80]] + [(j, "think") for j in pb_think[:: max(1, len(pb_think) // 6)][:6]]
    if not pts:
        continue
    sel = torch.tensor([len(pids) + j for j, _ in pts], device=m.device)
    z = hn[sel].float() @ E.float().T
    A = act["a"][sel].float().cpu().numpy(); Rin = res_h[0][sel].float().cpu().numpy()
    for k, (j, ph) in enumerate(pts):
        if ph == "think":
            y = "think"
        elif not fin:
            y = "stuck"
        elif gids[j + 1] == BS:
            y = "final"
        else:
            y = "healthy"
        X.append(A[k].astype(np.float16)); R.append(Rin[k].astype(np.float32)); Y.append(y); GID.append(pid); TPOS.append(j - start)
        GAP.append(float(z[k, BS] - z[k].max()))
    print(pid, "stuck" if not fin else "finished", len(pts), flush=True)
X = np.stack(X).astype(np.float32); R = np.stack(R); Y = np.array(Y); GID = np.array(GID); TPOS = np.array(TPOS); GAP = np.array(GAP)
np.savez(f"{tf.CACHE_DIR}/doubtprobe_{tag}.npz", X=X.astype(np.float16), R=R, Y=Y, GID=GID, TPOS=TPOS, GAP=GAP)
print("points:", {k: int((Y == k).sum()) for k in ["healthy", "stuck", "final", "think"]})
rng = np.random.default_rng(0)


def cv(F, y, g, k=5):
    ug = np.array(sorted(set(g))); perm = rng.permutation(len(ug)); folds = np.array_split(perm, k)
    accs = []
    for f in folds:
        tg = set(ug[f]); tr = np.array([i for i in range(len(y)) if g[i] not in tg]); te = np.array([i for i in range(len(y)) if g[i] in tg])
        if len(tr) == 0 or len(te) == 0:
            continue
        w = F[tr][y[tr] == 1].mean(0) - F[tr][y[tr] == 0].mean(0); b = 0.5 * (w @ F[tr][y[tr] == 1].mean(0) + w @ F[tr][y[tr] == 0].mean(0))
        pred = F[te] @ w > b; yt = y[te]
        accs.append(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))
    return float(np.mean(accs))


for name, F in [("MLP hidden activations (3584)", X), ("layer-23 input residual (1024)", R)]:
    s = (Y == "stuck") | (Y == "healthy")
    print(f"{name}: stuck vs healthy(non-final) paragraph breaks, CV by response: {cv(F[s], (Y[s] == 'stuck').astype(int), GID[s]):.3f}  (n stuck {int((Y[s]=='stuck').sum())}, healthy {int((Y[s]=='healthy').sum())})")
    s2 = (Y == "resp") | (Y == "think")
    print(f"{name}: response-phase vs thinking-phase paragraph breaks: {cv(F[s2], (Y[s2] == 'resp').astype(int), GID[s2]):.3f}")
print("box gap by class:", {k: (round(float(np.median(GAP[Y == k])), 1), round(float(np.quantile(GAP[Y == k], 0.9)), 1)) for k in ["healthy", "stuck", "final", "think"]})
