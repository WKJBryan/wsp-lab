"""Can 'the current token is a paragraph break' be read out linearly from the layer-23 MLP hidden
activations (so that a rank-one weight edit can act only there)? Collect activations at paragraph
breaks and at other positions, fit a difference-in-means readout, and report its selectivity.
Writes work/cache/pbreadout_<tag>.npz
Usage: python a52_pb_readout.py <model dir> <pass no> <tag> [n generations]"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

model_dir, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ngen = int(sys.argv[4]) if len(sys.argv) > 4 else 48
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(model_dir)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
V = E.shape[0]
TE, NL, NL2, BS = tf.T_THINK_END, 198, 271, 59
A, Y, PH, GID, RMS, LBOX = [], [], [], [], [], []
idx = np.linspace(0, len(rows) - 1, ngen).astype(int)
for i in idx:
    r = rows[i]
    pid = r["id"]
    gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"])
    te = gids.index(TE) if TE in gids else None
    end = min(len(gids), (te + 2500) if te is not None else 5000)
    ids = torch.tensor(pids + gids[: end + 1], device=m.device)
    act = {}; res_h = []
    hooks = [Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0].detach())),
             Lm.norm.register_forward_pre_hook(lambda mod, args: res_h.append(args[0][0].detach()))]
    with torch.no_grad():
        hn = tf.run(m, ids, keep="none")["hn"]
    for h in hooks:
        h.remove()
    T = len(ids)
    ispb = np.zeros(T, bool)
    for j in range(len(pids), T - 1):
        if gids[j - len(pids)] == NL2 or (gids[j - len(pids)] == NL and gids[j - 1 - len(pids)] == NL):
            ispb[j] = True
    other = np.zeros(T, bool); other[len(pids)::7] = True; other &= ~ispb
    sel = np.where(ispb | other)[0]
    st = torch.tensor(sel, device=m.device)
    A.append(act["a"][st].float().half().cpu().numpy())
    with torch.no_grad():
        rms = res_h[0][st].float().pow(2).mean(-1).sqrt().cpu().numpy()
        z = (hn[st].float() @ E.float().T)
        lse = torch.logsumexp(z, -1)
        LBOX.append((z[:, BS] - lse).cpu().numpy()); RMS.append(rms)
    g0 = len(pids)
    for j in sel:
        PH.append("prompt" if j < g0 else ("response" if (te is not None and j - g0 > te) else "thinking"))
    Y.append(ispb[sel]); GID.append(np.array([pid] * len(sel)))
    print(pid, int(ispb.sum()), int(other.sum()), flush=True)
A = np.concatenate(A).astype(np.float32); Y = np.concatenate(Y); PH = np.array(PH); GID = np.concatenate(GID)
RMS = np.concatenate(RMS); LBOX = np.concatenate(LBOX)
np.savez(f"{tf.CACHE_DIR}/pbreadout_{tag}.npz", A=A.astype(np.float16), Y=Y, PH=PH, GID=GID, RMS=RMS, LBOX=LBOX)
print("positions:", len(Y), "paragraph breaks:", int(Y.sum()), {p: int(((PH == p) & Y).sum()) for p in ["thinking", "response", "prompt"]})
rng = np.random.default_rng(0)
ug = np.array(sorted(set(GID))); perm = rng.permutation(len(ug)); folds = np.array_split(perm, 5)
score = np.zeros(len(Y))
for f in folds:
    tg = set(ug[f]); tr = np.isin(GID, list(tg))
    w = A[~tr][Y[~tr]].mean(0) - A[~tr][~Y[~tr]].mean(0)
    score[tr] = A[tr] @ w
gap = A[Y].mean(0) @ (A[Y].mean(0) - A[~Y].mean(0)) - A[~Y].mean(0) @ (A[Y].mean(0) - A[~Y].mean(0))
s = score / gap
print("readout score in units of the class gap (1.0 = mean at paragraph breaks, 0 = mean elsewhere), cross-validated:")
for name, sel in [("paragraph breaks (all)", Y), ("  in thinking", Y & (PH == "thinking")), ("  in response", Y & (PH == "response")),
                  ("other positions (all)", ~Y), ("  thinking", ~Y & (PH == "thinking")), ("  response", ~Y & (PH == "response")), ("  prompt", ~Y & (PH == "prompt"))]:
    if sel.sum():
        q = np.quantile(s[sel], [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"  {name:28s} n={int(sel.sum()):6d}  quantiles 5/25/50/75/95: {q.round(2)}")
thr = 0.5
print(f"at threshold 0.5: recall {float((s[Y] > thr).mean()):.3f}, false-positive rate elsewhere {float((s[~Y] > thr).mean()):.4f} "
      f"(thinking {float((s[~Y & (PH == 'thinking')] > thr).mean()):.4f}, response {float((s[~Y & (PH == 'response')] > thr).mean()):.4f})")
