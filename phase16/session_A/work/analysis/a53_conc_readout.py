"""Is 'the preceding sentence was a conclusion statement' linearly readable at the positions where
the response must decide between the final \boxed line and more text? Collect layer-23 MLP hidden
activations at labelled positions of the responses of two passes (and thinking-phase controls).
Writes work/cache/concreadout_<tag>.npz
Usage: python a53_conc_readout.py <model dir> <tag> <pass nos comma-separated>"""
import sys, json, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

model_dir, tag = sys.argv[1], sys.argv[2]
passes = [int(x) for x in sys.argv[3].split(",")]
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
m = tf.load(model_dir)
Lm = tf.lm(m)
E = m.get_input_embeddings().weight
TE, NL, NL2, BS = tf.T_THINK_END, 198, 271, 59
CONC = re.compile(r"(\banswer\b|^\s*therefore|^\s*thus|^\s*hence|^\s*so[,.]|\bso the\b|\bfinal\b|\bin conclusion\b|the value of .* is|the result is)", re.I)
DOUBT = re.compile(r"(\bwait\b|\bmaybe\b|^\s*if |\bcould\b|\bwould\b|\?\s*$|\bcheck\b|\bverify\b|\blet'?s\b|\bpossibly\b|\blikely\b|\bseems?\b|\bperhaps\b|not necessarily)", re.I)


def conclusion(line):
    return bool(CONC.search(line)) and not DOUBT.search(line)


A, Y, K, GID, TXT = [], [], [], [], []
seen = set()
for n in passes:
    for r in tf.gen_rows(n):
        if not r["closed"] or r["id"] in seen:
            continue
        seen.add(r["id"])
        pid = r["id"]
        gids = tok.encode(r["text"]); pids = tf.prompt_ids(pr[pid]["q"]); te = gids.index(TE); start = te + 1
        end = min(len(gids), start + 3500)
        ids = torch.tensor(pids + gids[: end + 1], device=m.device)
        act = {}
        h = Lm.layers[23].mlp.down_proj.register_forward_pre_hook(lambda mod, args: act.__setitem__("a", args[0][0].detach()))
        with torch.no_grad():
            tf.run(m, ids, keep="none")
        h.remove()
        pieces = [tok.decode([x]) for x in gids]
        # line index: for each response token j, the text of the current line up to j
        pts = {}
        line_start = start
        for j in range(start, end - 1):
            if gids[j] in (NL, NL2):
                line = "".join(pieces[line_start: j])
                if len(line.strip()) > 12 and (line.rstrip().endswith(".") or line.rstrip().endswith("$") or line.rstrip().endswith("}") or line.rstrip().endswith(")") or line.rstrip().endswith("]")):
                    if gids[j] == NL2 or gids[j - 1] == NL:
                        k = ("PB_conc" if conclusion(line) else "PB_other")
                    else:
                        k = ("NL_conc" if conclusion(line) else "NL_other")
                    pts.setdefault(k, []).append(j)
                line_start = j + 1
        # thinking controls: newline positions after conclusion-like lines
        th_c, th_o, ls = [], [], 5
        for j in range(5, te - 1):
            if gids[j] in (NL, NL2):
                line = "".join(pieces[ls: j])
                if len(line.strip()) > 12 and line.rstrip().endswith("."):
                    (th_c if conclusion(line) else th_o).append(j)
                ls = j + 1
        pts["TH_conc"] = th_c[:: max(1, len(th_c) // 3)][:3]
        pts["TH_other"] = th_o[:: max(1, len(th_o) // 3)][:3]
        pts["NL_other"] = pts.get("NL_other", [])[:: max(1, len(pts.get("NL_other", [])) // 8)][:8]
        pts["PB_other"] = pts.get("PB_other", [])[:: max(1, len(pts.get("PB_other", [])) // 8)][:8]
        pts["MID"] = list(range(start + 5, end - 1, 37))[:40]
        for k, js in pts.items():
            if not js:
                continue
            st = torch.tensor([len(pids) + j for j in js], device=m.device)
            A.append(act["a"][st].float().half().cpu().numpy()); Y.append(np.array([k == "PB_conc" or k == "NL_conc"] * len(js)))
            K.append(np.array([k] * len(js))); GID.append(np.array([f"{pid}_{n}"] * len(js))); TXT.append(np.array([""] * len(js)))
        print(n, pid, {k: len(v) for k, v in pts.items()}, flush=True)
A = np.concatenate(A).astype(np.float32); Y = np.concatenate(Y); K = np.concatenate(K); GID = np.concatenate(GID)
np.savez(f"{tf.CACHE_DIR}/concreadout_{tag}.npz", A=A.astype(np.float16), Y=Y, K=K, GID=GID)
print("positions by class:", {k: int((K == k).sum()) for k in sorted(set(K))})
rng = np.random.default_rng(0)
ug = np.array(sorted(set(GID))); perm = rng.permutation(len(ug)); folds = np.array_split(perm, 5)
# train on paragraph breaks and single newlines of the response only
trn = np.isin(K, ["PB_conc", "NL_conc", "PB_other", "NL_other"])
score = np.zeros(len(Y))
for f in folds:
    tg = set(ug[f]); test = np.isin(GID, list(tg)); tr = (~test) & trn
    w = A[tr][Y[tr]].mean(0) - A[tr][~Y[tr]].mean(0)
    score[test] = A[test] @ w
    score[test & ~trn] = A[test & ~trn] @ w
mu1, mu0 = A[trn & Y].mean(0), A[trn & ~Y].mean(0)
gap = float((mu1 - mu0) @ (mu1 - mu0))
s = score / gap
print("readout score (1 = mean at conclusion breaks, 0 = mean at other breaks), cross-validated by response:")
for k in sorted(set(K)):
    sel = K == k
    print(f"  {k:9s} n={int(sel.sum()):5d}  quantiles 10/25/50/75/90: {np.quantile(s[sel], [0.1, 0.25, 0.5, 0.75, 0.9]).round(2)}")
pb = np.isin(K, ["PB_conc", "PB_other"])
for thr in [0.3, 0.4, 0.5]:
    print(f"  at paragraph breaks, threshold {thr}: recall {float((s[pb & Y] > thr).mean()):.3f}, false-positive rate at other paragraph breaks {float((s[pb & ~Y] > thr).mean()):.3f}, "
          f"at thinking conclusion-like newlines {float((s[K == 'TH_conc'] > thr).mean()) if (K == 'TH_conc').any() else float('nan'):.3f}, mid-line {float((s[K == 'MID'] > thr).mean()):.4f}")
