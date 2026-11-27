"""Verify an edited checkpoint against the base with teacher-forced reads on baseline trajectories.
Reports: </think> probability at line ends (base vs edit), change of the top competitor, change of
the probability of the tokens the model actually wrote (all positions, split by line-end / other),
and the largest change of any other token's logit.
Usage: python verify_edit.py <cand dir> [pass no] [n generations] [max prefix tokens]"""
import sys, json, os
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

cand = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ngen = int(sys.argv[3]) if len(sys.argv) > 3 else 16
maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 3500
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
loops = {d["id"]: d for d in json.load(open(f"{tf.CACHE_DIR}/loops_pass{n}.json"))}
tok = tf.tokenizer()
TE = tf.T_THINK_END
mb = tf.load(tf.BASE_DIR)
mc = tf.load(cand)
E = mb.get_input_embeddings().weight
V = E.shape[0]
nl_ids = {tok.encode(s, add_special_tokens=False)[0] for s in ["\n", "\n\n"]}
# which tensors differ
from safetensors import safe_open
with safe_open(os.path.join(tf.BASE_DIR, "model.safetensors-00001-of-00001.safetensors"), framework="pt", device="cpu") as fb, \
     safe_open(os.path.join(cand, "model.safetensors-00001-of-00001.safetensors"), framework="pt", device="cpu") as fc:
    changed = [k for k in fb.keys() if not torch.equal(fb.get_tensor(k), fc.get_tensor(k))]
    added = [k for k in fc.keys() if k not in list(fb.keys())]
print("tensors that differ from the base:", changed, "| added:", added)
del mb, mc
torch.cuda.empty_cache()

picks = [rows[i] for i in np.linspace(0, len(rows) - 1, ngen).astype(int)]
for name, path in [("base", tf.BASE_DIR), ("edit", cand)]:
    m = tf.load(path)
    agg = dict(le_p=[], le_pact=[], le_top=[], oth_pact=[], oth_dte=[], first_p=[], first_ptop=[])
    for r in picks:
        pid = r["id"]
        gids = tok.encode(r["text"])
        pids = tf.prompt_ids(pr[pid]["q"])
        lim = loops[pid].get("loop_start", len(gids)) if (pid in loops and loops[pid].get("period")) else (gids.index(TE) if TE in gids else len(gids))
        G = min(lim, maxlen, len(gids) - 1)
        ids = torch.tensor(pids + gids[:G], device=m.device)
        with torch.no_grad():
            hn = tf.run(m, ids, keep="none")["hn"]
            out = []
            for a in range(0, G, 512):
                b = min(G, a + 512)
                z = hn[len(pids) - 1 + a: len(pids) - 1 + b].float() @ E.float().T
                seen = torch.zeros(b - a, V, dtype=torch.bool, device=z.device)
                for i in range(a, b):
                    seen[i - a, gids[max(0, i - 20000): i]] = True
                padj = tf.adj_sample_prob(z, seen=seen)
                praw = torch.softmax(z, -1)
                nxt = torch.tensor(gids[a:b], device=z.device)
                zc = z.clone(); zc.scatter_(1, nxt[:, None], -1e9)
                out.append(torch.stack([padj[:, TE], padj.gather(1, nxt[:, None]).squeeze(1), padj.max(-1).values,
                                        praw[:, TE], praw.gather(1, nxt[:, None]).squeeze(1).log(), z.max(-1).values, zc.max(-1).values], 1))
        A = torch.cat(out).cpu().numpy()      # [G, 7]: pTE, p_next(adj), p_top1(adj), pTE raw, log p_next raw, top1 logit, 2nd logit
        np.savez(f"{tf.CACHE_DIR}/verify_{name}_{pid}.npz", A=A)
        isle = np.array([False] + [g in nl_ids for g in gids[: G - 1]])     # prediction right after a newline token
        agg["le_p"].append(A[isle, 0]); agg["le_pact"].append(A[isle, 1]); agg["le_top"].append(A[isle, 2])
        agg["oth_pact"].append(A[~isle, 4]); agg["oth_dte"].append(A[~isle, 3])
        agg["first_p"].append(A[0, 0]); agg["first_ptop"].append(A[0, 2])
    cat = {k: np.concatenate([np.atleast_1d(x) for x in v]) for k, v in agg.items()}
    print(f"[{name}] line ends: n={cat['le_p'].size}, mean p(</think>)={cat['le_p'].mean():.4f}, frac(p>0.5)={float((cat['le_p'] > 0.5).mean()):.4f}, "
          f"frac(p>0.05)={float((cat['le_p'] > 0.05).mean()):.4f}, median p(top1)={np.median(cat['le_top']):.3f} | other positions: mean log p(written token)={cat['oth_pact'].mean():.4f}, "
          f"max raw p(</think>)={cat['oth_dte'].max():.2e}, frac(raw p>1e-3)={float((cat['oth_dte'] > 1e-3).mean()):.5f} | first token: mean p(</think>)={cat['first_p'].mean():.4f} max {cat['first_p'].max():.3f}")
    del m
    torch.cuda.empty_cache()
# paired comparison
dl_p, dl_act, d_o_act, d_top2 = [], [], [], []
for r in picks:
    a = np.load(f"{tf.CACHE_DIR}/verify_base_{r['id']}.npz")["A"]; b = np.load(f"{tf.CACHE_DIR}/verify_edit_{r['id']}.npz")["A"]
    gids = tok.encode(r["text"]); G = len(a)
    isle = np.array([False] + [g in nl_ids for g in gids[: G - 1]])
    dl_p.append(b[isle, 0] - a[isle, 0]); dl_act.append(b[isle, 1] - a[isle, 1]); d_o_act.append(b[~isle, 4] - a[~isle, 4]); d_top2.append(b[isle, 6] - a[isle, 6])
dl_p, dl_act, d_o_act, d_top2 = map(np.concatenate, (dl_p, dl_act, d_o_act, d_top2))
print(f"\npaired (edit - base): line ends: mean d p(</think>)={dl_p.mean():+.4f}, mean d p(written token)={dl_act.mean():+.4f}; "
      f"other positions: mean d log p(written)={d_o_act.mean():+.5f} (min {d_o_act.min():+.3f}); second-best logit at line ends: max |change| {np.abs(d_top2).max():.3f}")
