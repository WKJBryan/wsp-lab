"""Linear probes on the residual stream at decision points (a22 cache): does the network
internally separate 'answer stated' from 'still deriving' at line ends? And what drives the
termination head (L22 GDN head 13)?
Usage: python a23_probes.py <tag>"""
import sys
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

tag = sys.argv[1]
store = torch.load(f"{tf.CACHE_DIR}/probe_{tag}.pt", weights_only=False)
tok = tf.tokenizer()
# assemble
X, y, gen, tpos, lab = [], [], [], [], []
Z13, C13, LTE, TOP1 = [], [], [], []
for g in store:
    for k, l in enumerate(g["labels"]):
        X.append(g["H"][k]); gen.append(g["id"]); lab.append(l); tpos.append(g["t"][k])
        Z13.append(g["z22"][k][13]); C13.append(g["core22"][k][13]); LTE.append(g["logit_te"][k]); TOP1.append(g["top1"][k])
X = np.stack(X).astype(np.float32)             # [N, 25, D]
Xnp = X
gen = np.array(gen); lab = np.array(lab); tpos = np.array(tpos); Z13 = np.stack(Z13); C13 = np.stack(C13)
LTE = np.array(LTE); TOP1 = np.array(TOP1)
print("points:", len(lab), dict(zip(*np.unique(lab, return_counts=True))), "generations:", len(set(gen)))
print("mean logit(</think>) by label:", {l: round(float(LTE[lab == l].mean()), 2) for l in ["close", "c1", "c0"]},
      "| mean top-1 logit:", {l: round(float(TOP1[lab == l].mean()), 2) for l in ["close", "c1", "c0"]})
rng = np.random.default_rng(0)
ug = np.array(sorted(set(gen)))

def cv_probe(Xa, ya, groups, k=5, ridge=0.0):
    """difference-in-means probe, cross-validated over groups. Returns mean balanced accuracy."""
    accs = []
    ug = np.array(sorted(set(groups))); perm = rng.permutation(len(ug)); folds = np.array_split(perm, k)
    for f in folds:
        test_g = set(ug[f]); tr = np.array([i for i in range(len(ya)) if groups[i] not in test_g], dtype=int); te = np.array([i for i in range(len(ya)) if groups[i] in test_g], dtype=int)
        if len(te) == 0:
            continue
        A, B = Xa[tr][ya[tr] == 1], Xa[tr][ya[tr] == 0]
        w = A.mean(0) - B.mean(0)
        b = 0.5 * (w @ A.mean(0) + w @ B.mean(0))
        pred = (Xa[te] @ w > b).astype(int)
        yt = ya[te]
        accs.append(0.5 * ((pred[yt == 1] == 1).mean() + (pred[yt == 0] == 0).mean()))
    return float(np.mean(accs))

print("\n=== P1: 'answer/conclusion stated' (c1) vs other line ends (c0), per layer, 5-fold CV by generation ===")
sel = (lab == "c1") | (lab == "c0")
ya = (lab[sel] == "c1").astype(int); ga = gen[sel]
print("  n c1:", int(ya.sum()), "n c0:", int((1 - ya).sum()))
line = []
for l in range(25):
    acc = cv_probe(X[sel][:, l], ya, ga)
    line.append(f"L{l}:{acc:.3f}")
print("  " + " ".join(line))
print("\n=== P2: closing points (close) vs all other line ends, per layer ===")
sel2 = (lab == "close") | (lab == "c0") | (lab == "c1")
# balance: use each close point with its own generation's other points -> leave-one-generation-out
ya2 = (lab[sel2] == "close").astype(int); ga2 = gen[sel2]
line = []
for l in range(25):
    acc = cv_probe(X[sel2][:, l], ya2, ga2, k=7)
    line.append(f"L{l}:{acc:.3f}")
print("  n close:", int(ya2.sum()), "others:", int((1 - ya2).sum()))
print("  " + " ".join(line))
print("\n=== P3: position in the generation (early < 1200 tokens vs late > 3000) at line ends, per layer ===")
sel3 = ((lab == "c1") | (lab == "c0")) & ((tpos < 1200) | (tpos > 3000))
ya3 = (tpos[sel3] > 3000).astype(int); ga3 = gen[sel3]
line = []
for l in range(25):
    acc = cv_probe(X[sel3][:, l], ya3, ga3)
    line.append(f"L{l}:{acc:.3f}")
print("  n early:", int((1 - ya3).sum()), "late:", int(ya3.sum()))
print("  " + " ".join(line))
print("\n=== P4: c1 vs c0 within the early phase only (t < 1500) ===")
sel4 = ((lab == "c1") | (lab == "c0")) & (tpos < 1500)
ya4 = (lab[sel4] == "c1").astype(int); ga4 = gen[sel4]
print("  n c1:", int(ya4.sum()), "n c0:", int((1 - ya4).sum()), "| acc L16 %.3f L20 %.3f L22 %.3f L23 %.3f L24 %.3f" % tuple(cv_probe(X[sel4][:, l], ya4, ga4) for l in [16, 20, 22, 23, 24]))

print("\n=== termination head: L22 GDN head 13 internals by label ===")
m = tf.load()
Lm = tf.lm(m)
Wo = Lm.layers[22].linear_attn.out_proj.weight.float().detach()
E = m.get_input_embeddings().weight.float().detach(); nw = Lm.norm.weight.float().detach()
ute = (1 + nw) * E[tf.T_THINK_END]
wo = Wo[:, 13 * 128:(13 + 1) * 128]                      # [D, 128]
al = wo @ ute                                             # alignment of each of the 128 gate dims with </think>
print("  |alignment| of head-13 output dims with </think>: max %.3f, mean %.3f; top dims:" % (float(al.abs().max()), float(al.abs().mean())),
      [(int(i), round(float(al[i]), 3)) for i in al.abs().topk(8).indices.tolist()])
for l in ["close", "c1", "c0"]:
    s = lab == l
    print(f"  {l:5s} n={int(s.sum()):4d} mean gate z (silu input): mean={Z13[s].mean():+.3f}  frac>0={float((Z13[s] > 0).float().mean()):.3f}  | core rms={float(np.sqrt((C13[s] ** 2).sum(-1).mean())):.2f}")
# which gate dims are most 'closing-specific'?
mu_c, mu_1, mu_0 = Z13[lab == "close"].mean(0), Z13[lab == "c1"].mean(0), Z13[lab == "c0"].mean(0)
sd = Z13.std(0) + 1e-6
d = (mu_c - mu_0) / sd
idx = np.argsort(-np.abs(d))[:10]
print("  gate dims most different at closing points (dim, z_close, z_c1, z_c0, alignment):")
for i in idx:
        print(f"    d{int(i):3d}: close {mu_c[i]:+.2f}  c1 {mu_1[i]:+.2f}  c0 {mu_0[i]:+.2f}  (d'={d[i]:+.2f})  align={float(al[i]):+.3f}")
# approximate contribution of head 13 = sum_d out_proj[:,d] * w_norm[d] * silu(z_d) * core_d  ... needs the normed core; report corr(silu(z_d)*core_d, logitTE)
import torch.nn.functional as F
g = torch.tensor(Z13); c = torch.tensor(C13)
act = (F.silu(g) * c).numpy()        # [N, 128] (up to the per-head RMS norm and norm.weight)
corr = [(int(i), float(np.corrcoef(act[:, i], LTE)[0, 1])) for i in range(128)]
corr.sort(key=lambda x: -abs(x[1]))
print("  head-13 dims whose (silu(z)*core) correlates most with the </think> logit:", [(i, round(c, 2)) for i, c in corr[:10]])
