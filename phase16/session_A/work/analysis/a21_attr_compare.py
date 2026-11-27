"""Compare the </think>-logit attribution between closing and continue points (a20 cache).
Usage: python a21_attr_compare.py <tag>"""
import sys, re
import numpy as np, torch
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

tag = sys.argv[1]
out = torch.load(f"{tf.CACHE_DIR}/attr_{tag}.pt", weights_only=False)
tok = tf.tokenizer()
groups = {}
for o in out:
    groups.setdefault(o["label"], []).append(o)
labels = [l for l in ["close", "cont_c", "cont"] if l in groups]
print("n per group:", {l: len(groups[l]) for l in labels})
print("\nmean </think> logit per group:", {l: round(float(np.mean([o["logit_te"] for o in groups[l]])), 2) for l in labels})
print("mean top1 logit per group  :", {l: round(float(np.mean([o["top"][0][1] for o in groups[l]])), 2) for l in labels})
print("\n=== mean contribution per layer (mixer / mlp) to the </think> logit ===")
print(f"{'layer':>5s} " + " ".join(f"{l:>22s}" for l in labels) + "   | close - cont_c")
tot = {l: dict(mix=np.zeros(24), mlp=np.zeros(24), head=np.zeros((24, 16)), neuron=[]) for l in labels}
for lay in range(24):
    line = f"L{lay:<4d} "
    vals = {}
    for l in labels:
        mx = np.mean([float(o["contrib"][(lay, "mixer")]) for o in groups[l]])
        mp = np.mean([float(o["contrib"][(lay, "mlp")]) for o in groups[l]])
        vals[l] = (mx, mp)
        line += f"   {mx:+7.2f} / {mp:+7.2f} "
    dc = (vals["close"][0] - vals["cont_c"][0], vals["close"][1] - vals["cont_c"][1]) if "cont_c" in vals else (0, 0)
    line += f"   | {dc[0]:+6.2f} / {dc[1]:+6.2f}"
    print(line)
for l in labels:
    print(f"\n=== {l}: sum mixer={sum(float(np.mean([float(o['contrib'][(lay, 'mixer')]) for o in groups[l]])) for lay in range(24)):+.2f} "
          f"sum mlp={sum(float(np.mean([float(o['contrib'][(lay, 'mlp')]) for o in groups[l]])) for lay in range(24)):+.2f}")

print("\n=== heads: mean contribution, close vs cont_c (heads with |difference| > 0.25) ===")
for l in labels:
    H = np.stack([np.stack([np.pad(o["contrib"][(lay, "mixer", "head")].numpy(), (0, 16 - len(o["contrib"][(lay, "mixer", "head")]))) for lay in range(24)]) for o in groups[l]])   # [n, 24, 16]
    tot[l]["head"] = H
if "close" in groups and "cont_c" in groups:
    Hc, Hk = tot["close"]["head"].mean(0), tot["cont_c"]["head"].mean(0)
    sd = tot["cont_c"]["head"].std(0) + 1e-6
    dif = Hc - Hk
    idx = [(abs(dif[l, h]), l, h) for l in range(24) for h in range(Hc.shape[1]) if (h < 8 or l % 4 != 3)]
    idx.sort(reverse=True)
    for _, l, h in idx[:25]:
        print(f"  L{l:02d} {'attn' if l % 4 == 3 else 'gdn '} head {h:2d}: close {Hc[l, h]:+.2f}  cont_c {Hk[l, h]:+.2f}  diff {dif[l, h]:+.2f}  (cont_c sd {sd[l, h]:.2f})")
    print("\n=== MLP neurons: largest close-vs-cont_c differences of contribution (mean over positions) ===")
    nc = np.stack([np.stack([o["contrib"][(lay, "mlp", "neuron")].numpy() for lay in range(24)]) for o in groups["close"]]).mean(0)     # [24, 3584]
    nk = np.stack([np.stack([o["contrib"][(lay, "mlp", "neuron")].numpy() for lay in range(24)]) for o in groups["cont_c"]]).mean(0)
    sk = np.stack([np.stack([o["contrib"][(lay, "mlp", "neuron")].numpy() for lay in range(24)]) for o in groups["cont_c"]]).std(0)
    d = nc - nk
    idx = [(abs(d[l, j]), l, j) for l in range(24) for j in range(3584)]
    idx.sort(reverse=True)
    print("  top 25:", ", ".join(f"L{l}n{j}({nc[l, j]:+.2f}/{nk[l, j]:+.2f})" for _, l, j in idx[:25]))
    print("  sum of top-25 diffs: %+.2f ; total neuron-sum diff: %+.2f" % (sum(d[l, j] for _, l, j in idx[:25]), d.sum()))
    print("  per layer: sum |diff| of neurons:", [round(float(np.abs(d[l]).sum()), 1) for l in range(24)])
    print("  per layer: sum diff:", [round(float(d[l].sum()), 2) for l in range(24)])

print("\n=== what follows at continue points (top-1 token) and the line before ===")
for l in labels:
    c = {}
    for o in groups[l]:
        w = tok.decode([o["top"][0][0]])
        c[w] = c.get(w, 0) + 1
    print(l, sorted(c.items(), key=lambda kv: -kv[1])[:6])
