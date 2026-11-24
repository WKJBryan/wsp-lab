"""Phase-7 atlas, part 2 — analysis of the activation probe (CPU). Implements exactly the
A1-A4 plan in phase7/protocol.md (addendum, committed before probing).

Inputs: phase7/probe_out/*.npz (from probe_activations.py), phase7/sets.json
Outputs: phase7/atlas_activation.json, phase7/fig_a1_layers.png, fig_a2_neurons.png, fig_a3_onset.png

Usage: uv run python phase7/analyze_activations.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "probe_out"
sets = json.loads((HERE / "sets.json").read_text())
items = sets["items"]
R = np.load(OUT / "resid.npz")
NZ = np.load(OUT / "neurons.npz")
TL = np.load(OUT / "timelines.npz")
ED = np.load(OUT / "edit.npz")
resid = {k: R[f"resid_{k}"].astype(np.float32) for k in ("pre", "loop", "clean", "deep")}
meta = {k: R[f"meta_{k}"] for k in resid}  # cols: item index, half, position
r3 = lambda x: round(float(x), 3)


def auc(x, y):
    """P(random x > random y)."""
    z = np.concatenate([x, y])
    ranks = np.empty(len(z)); ranks[np.argsort(z)] = np.arange(1, len(z) + 1)
    return float((ranks[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def unit(v):
    return v / np.linalg.norm(v)


def gen_means(region, proj):
    """Mean projection per generation -> {item index: mean}."""
    idx = meta[region][:, 0]
    return {i: proj[idx == i].mean() for i in np.unique(idx)}


half = {k: meta[k][:, 1] for k in resid}
res = dict(n_positions={k: int(len(v)) for k, v in resid.items()})

# ---------- A1: loop direction per layer ----------
a1 = []
for l in range(25):
    X = {k: v[:, l] for k, v in resid.items()}
    d = unit(X["loop"][half["loop"] == 0].mean(0) - X["pre"][half["pre"] == 0].mean(0))
    d_depth = unit(X["loop"][half["loop"] == 0].mean(0) - X["deep"][half["deep"] == 0].mean(0))
    ho = {k: X[k][half[k] == 1] @ d for k in X}
    gl, gp = gen_means("loop", X["loop"] @ d), gen_means("pre", X["pre"] @ d)
    held = [i for i in gl if items[i]["half"] == 1]
    a1.append(dict(
        layer_out=l - 1,  # -1 = embeddings; L = residual after layer L
        cos_u15=r3(d @ ED["u15"]), cos_u19=r3(d @ ED["u19"]), cos_depth_dir=r3(d @ d_depth),
        auc_loop_vs_pre=r3(auc(ho["loop"], ho["pre"])), auc_loop_vs_deep=r3(auc(ho["loop"], ho["deep"])),
        auc_loop_vs_clean=r3(auc(ho["loop"], ho["clean"])),
        paired_win_rate=r3(np.mean([gl[i] > gp[i] for i in held])),
        auc_u15_loop_vs_pre=r3(auc(X["loop"] @ ED["u15"], X["pre"] @ ED["u15"])),
        auc_u15_loop_vs_deep=r3(auc(X["loop"] @ ED["u15"], X["deep"] @ ED["u15"])),
    ))
res["A1_loop_direction"] = a1

# ---------- A2: neurons ----------
def stats(region, halves=(0, 1)):
    n = sum(int(NZ[f"cnt_{region}_{h}"]) for h in halves)
    s = sum(NZ[f"sum_{region}_{h}"] for h in halves)
    q = sum(NZ[f"sq_{region}_{h}"] for h in halves)
    m = s / n
    return m, np.maximum(q / n - m * m, 1e-12)


def cohen(a, b, halves=(0, 1)):
    (ma, va), (mb, vb) = stats(a, halves), stats(b, halves)
    return (ma - mb) / np.sqrt((va + vb) / 2)  # (24, 3584)


D = cohen("loop", "pre")
D_deep = cohen("loop", "deep")
D0, D1 = cohen("loop", "pre", (0,)), cohen("loop", "pre", (1,))
a2 = dict(per_layer=[], edited={})
for L in range(24):
    a2["per_layer"].append(dict(layer=L, max_abs_d=r3(np.abs(D[L]).max()), n_abs_d_gt_0p8=int((np.abs(D[L]) > 0.8).sum()),
                                split_half_r=r3(np.corrcoef(D0[L], D1[L])[0, 1]),
                                top5=[int(j) for j in np.argsort(-np.abs(D[L]))[:5]]))
for L in (15, 19):
    cols = ED[f"cols{L}"]
    for tag, dd in (("loop_vs_pre", D[L]), ("loop_vs_deep", D_deep[L])):
        order = np.argsort(-np.abs(dd))
        pct = 1 - np.argsort(order)[cols] / len(dd)  # 1.0 = strongest neuron in the layer
        a2["edited"][f"L{L}_{tag}"] = dict(
            top64_overlap=int(len(set(order[:64].tolist()) & set(cols.tolist()))), chance_overlap=1.14,
            edited_median_percentile=r3(np.median(pct)), edited_median_abs_d=r3(np.median(np.abs(dd[cols]))),
            all_median_abs_d=r3(np.median(np.abs(dd))),
            sign_agreement_with_a=r3(np.mean(np.sign(dd[cols]) == np.sign(ED[f"a{L}"]))))
res["A2_neurons"] = a2

# ---------- A3: selectivity of the original edit (drive along u) ----------
a3 = {}
for L in (15, 19):
    tl = lambda kind, it: TL[f"{kind}{L}_{it['id']}_{it['group']}"].astype(np.float32)
    at = {k: np.array([tl("drive", items[i])[p] for i, _, p in meta[k]]) for k in resid}
    proj_sd = float(np.concatenate([tl("proj", it) for it in items if it["group"] == "clean"]).std())
    a3[f"L{L}"] = dict(
        mean_drive={k: r3(v.mean()) for k, v in at.items()}, sd_drive={k: r3(v.std()) for k, v in at.items()},
        resid_sd_along_u_clean=r3(proj_sd),
        auc_loop_vs_pre=r3(auc(at["loop"], at["pre"])), auc_loop_vs_deep=r3(auc(at["loop"], at["deep"])),
        auc_loop_vs_clean=r3(auc(at["loop"], at["clean"])), auc_pre_vs_clean=r3(auc(at["pre"], at["clean"])))
    edges = np.arange(-2000, 1501, 250)
    for kind in ("drive", "proj"):
        rows = []
        for it in (x for x in items if x["group"] == "stuck"):
            s, on = tl(kind, it), it["onset"]
            rows.append([s[on + a:on + b].mean() if on + a >= 0 else np.nan for a, b in zip(edges[:-1], edges[1:])])
        rows = np.array(rows)
        a3[f"L{L}"][f"onset_curve_{kind}"] = dict(
            bin_start=edges[:-1].tolist(), mean=[r3(x) for x in np.nanmean(rows, 0)],
            sem=[r3(x) for x in np.nanstd(rows, 0) / np.sqrt((~np.isnan(rows)).sum(0))])
res["A3_edit_selectivity"] = a3

# ---------- A4: foreseeability (exploratory; difficulty-confounded) ----------
a4 = []
early = {k: meta[k][:, 2] < 1500 for k in ("pre", "clean")}
for l in range(25):
    X = {k: resid[k][:, l] for k in ("pre", "clean")}
    f = {k: early[k] & (half[k] == 0) for k in X}
    h = {k: early[k] & (half[k] == 1) for k in X}
    d = unit(X["pre"][f["pre"]].mean(0) - X["clean"][f["clean"]].mean(0))
    gp = [X["pre"][h["pre"] & (meta["pre"][:, 0] == i)] @ d for i in np.unique(meta["pre"][h["pre"], 0])]
    gc = [X["clean"][h["clean"] & (meta["clean"][:, 0] == i)] @ d for i in np.unique(meta["clean"][h["clean"], 0])]
    a4.append(dict(layer_out=l - 1, auc_positions=r3(auc(X["pre"][h["pre"]] @ d, X["clean"][h["clean"]] @ d)),
                   auc_generations=r3(auc(np.array([g.mean() for g in gp]), np.array([g.mean() for g in gc]))),
                   n_gen=[len(gp), len(gc)]))
res["A4_foreseeability"] = a4

(HERE / "atlas_activation.json").write_text(json.dumps(res, indent=1))

# ---------- plots ----------
lay = [r["layer_out"] for r in a1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(lay, [abs(r["cos_u15"]) for r in a1], label="|cos(d, u15)|")
ax[0].plot(lay, [abs(r["cos_u19"]) for r in a1], label="|cos(d, u19)|")
ax[0].axhline(1 / np.sqrt(1024), ls=":", c="gray", label="chance")
ax[0].set(xlabel="residual after layer", ylabel="|cosine|", title="A1: activation loop direction vs the edit's direction")
ax[0].legend()
for key, lab in (("auc_loop_vs_pre", "d: loop vs pre"), ("auc_loop_vs_deep", "d: loop vs deep"),
                 ("auc_u15_loop_vs_pre", "u15: loop vs pre"), ("auc_u15_loop_vs_deep", "u15: loop vs deep")):
    ax[1].plot(lay, [r[key] for r in a1], label=lab)
ax[1].plot(lay, [r["auc_positions"] for r in a4], ls="--", label="A4 early: to-be-stuck vs clean")
ax[1].axhline(0.5, ls=":", c="gray"); ax[1].axhline(0.75, ls=":", c="k")
ax[1].set(xlabel="residual after layer", ylabel="held-out AUC", title="Separation (0.75 line = pre-registered 'separates')")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "fig_a1_layers.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, L in enumerate((15, 19)):
    cols = ED[f"cols{L}"]
    ax[k].hist(D[L], bins=80, color="lightgray", label="all 3,584 neurons")
    ax[k].plot(D[L][cols], np.full(len(cols), 5), "r|", ms=18, label="64 edited columns")
    ax[k].set(xlabel="effect size, loop vs pre (Cohen's d)", title=f"A2: L{L} neurons", yscale="log"); ax[k].legend()
ax[2].bar(range(24), [r["n_abs_d_gt_0p8"] for r in a2["per_layer"]])
ax[2].set(xlabel="layer", ylabel="neurons with |d| > 0.8", title="Where loop-vs-pre differences live")
fig.tight_layout(); fig.savefig(HERE / "fig_a2_neurons.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, kind in enumerate(("drive", "proj")):
    for L in (15, 19):
        c = a3[f"L{L}"][f"onset_curve_{kind}"]
        x, m, s = np.array(c["bin_start"]) + 125, np.array(c["mean"]), np.array(c["sem"])
        ax[k].plot(x, m, label=f"L{L}"); ax[k].fill_between(x, m - s, m + s, alpha=0.25)
        if kind == "drive":
            ax[k].axhline(a3[f"L{L}"]["mean_drive"]["clean"], ls=":", c=f"C{(15, 19).index(L)}")
    ax[k].axvline(0, c="k", ls="--")
    ax[k].set(xlabel="tokens relative to loop onset", ylabel="edit drive s(t)" if kind == "drive" else "<resid, u>",
              title=f"A3: {kind} around loop onset (dotted = clean mean)" if kind == "drive" else "A3: residual along u")
    ax[k].legend()
fig.tight_layout(); fig.savefig(HERE / "fig_a3_onset.png", dpi=130); plt.close(fig)

# ---------- console summary ----------
best = max(a1, key=lambda r: r["auc_loop_vs_deep"])
print("positions:", res["n_positions"])
print("A1 best layer (loop vs deep, held-out):", best)
print("A1 at L15/L19 outputs:", a1[16], a1[20])
print("A2 edited columns:", json.dumps(a2["edited"], indent=1))
print("A2 layers with most |d|>0.8:", sorted(a2["per_layer"], key=lambda r: -r["n_abs_d_gt_0p8"])[:5])
for L in (15, 19):
    print(f"A3 L{L}:", {k: v for k, v in a3[f"L{L}"].items() if not k.startswith("onset")})
    print(f"   onset curve drive:", a3[f"L{L}"]["onset_curve_drive"]["mean"])
print("A4 best layer:", max(a4, key=lambda r: r["auc_generations"]))
