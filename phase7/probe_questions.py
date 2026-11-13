"""Phase-7 atlas, part 3 — question-type map: what lights up while the stock model READS
different questions (forward pass over the prompt only; no generation).

Implements Q1-Q4 of the part-3 addendum in phase7/protocol.md. Needs part 2's
phase7/probe_out/{resid,neurons}.npz for the loop direction / loop neurons (Q4).

Outputs: phase7/atlas_questions.json, fig_q1_decoding.png, fig_q2_neurons.png
         (raw activations cached in phase7/probe_out/questions.npz, git-ignored)

Usage: uv run python phase7/probe_questions.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "probe_out"
CACHE = OUT / "questions.npz"
r3 = lambda x: round(float(x), 3)


def probe():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stock = ROOT / "phase3" / "models" / "stock"
    tok = AutoTokenizer.from_pretrained(stock)
    model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.bfloat16).eval().to("cuda")
    down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
            if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, 3584)}
    assert sorted(down) == list(range(24))
    cur = {}
    for L, m in down.items():
        m.register_forward_pre_hook(lambda _, inp, L=L: cur.__setitem__(L, inp[0][0].float().mean(0)))
    problems = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))
    last, mean, neur = [], [], []
    for p in problems:
        prompt = tok.apply_chat_template([{"role": "user", "content": p["q"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        with torch.no_grad():
            hs = model(tok(prompt, return_tensors="pt").input_ids.to("cuda"), output_hidden_states=True).hidden_states
        last.append(torch.stack([h[0, -1].float() for h in hs]).cpu().numpy())
        mean.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy())
        neur.append(torch.stack([cur[L] for L in range(24)]).cpu().numpy())
    OUT.mkdir(exist_ok=True)
    np.savez(CACHE, last=np.array(last, np.float32), mean=np.array(mean, np.float32), neurons=np.array(neur, np.float16))


def auc(x, y):
    z = np.concatenate([x, y])
    ranks = np.empty(len(z)); ranks[np.argsort(z)] = np.arange(1, len(z) + 1)
    return float((ranks[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def unit(v):
    return v / np.linalg.norm(v)


def main():
    if not CACHE.exists():
        probe()
    Z = np.load(CACHE)
    from datasets import load_from_disk
    ds = load_from_disk(str(ROOT / "phase3" / "data" / "math500"))
    subjects = sorted(set(ds["subject"]))
    subj = np.array([subjects.index(s) for s in ds["subject"]])
    level = np.array(ds["level"])
    half = np.arange(500) % 2
    fit, test = half == 0, half == 1
    res = dict(subjects=subjects, majority_baseline=r3(np.bincount(subj[test]).max() / test.sum()))

    # ---------- Q1 subject decoding / Q2 difficulty ----------
    q1, q2, diff_dir = {}, {}, {}
    for readout in ("last", "mean"):
        X = Z[readout]  # (500, 25, 1024)
        accs, aucs, confs = [], [], []
        for l in range(25):
            x = X[:, l]
            mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
            z = (x - mu) / sd
            cent = np.stack([unit(z[fit & (subj == s)].mean(0)) for s in range(len(subjects))])
            pred = (z[test] / (np.linalg.norm(z[test], axis=1, keepdims=True) + 1e-9) @ cent.T).argmax(1)
            accs.append(r3((pred == subj[test]).mean()))
            confs.append(pred)
            d = unit(x[fit & (level >= 4)].mean(0) - x[fit & (level <= 2)].mean(0))
            diff_dir[(readout, l)] = d
            aucs.append(r3(auc(x[test & (level >= 4)] @ d, x[test & (level <= 2)] @ d)))
        best = int(np.argmax(accs))
        per_subject = {s: r3((confs[best][subj[test] == k] == k).mean()) for k, s in enumerate(subjects)}
        q1[readout] = dict(accuracy_by_layer_out=dict(zip(range(-1, 24), accs)), best_layer_out=best - 1,
                           best_accuracy=accs[best], embedding_layer_accuracy=accs[0], per_subject_at_best=per_subject)
        q2[readout] = dict(auc_by_layer_out=dict(zip(range(-1, 24), aucs)), best_layer_out=int(np.argmax(aucs)) - 1,
                           best_auc=max(aucs))
    res["Q1_subject"], res["Q2_difficulty"] = q1, q2

    # ---------- Q3 subject-selective neurons ----------
    N = Z["neurons"].astype(np.float32)  # (500, 24, 3584)

    def eta2(mask):
        x, s = N[mask], subj[mask]
        grand = x.mean(0)
        between = sum((s == k).sum() * (x[s == k].mean(0) - grand) ** 2 for k in range(len(subjects)))
        return between / (((x - grand) ** 2).sum(0) + 1e-12)  # (24, 3584)

    E, E0, E1 = eta2(np.ones(500, bool)), eta2(fit), eta2(test)
    res["Q3_neurons"] = [dict(layer=L, n_eta2_gt_0p3=int((E[L] > 0.3).sum()), max_eta2=r3(E[L].max()),
                              split_half_r=r3(np.corrcoef(E0[L], E1[L])[0, 1]),
                              top5=[int(j) for j in np.argsort(-E[L])[:5]]) for L in range(24)]

    # ---------- Q4 relation to the loop map ----------
    R = np.load(OUT / "resid.npz")
    loop_dir = [unit(R["resid_loop"][:, l].astype(np.float32).mean(0) - R["resid_pre"][:, l].astype(np.float32).mean(0))
                for l in range(25)]
    NZ = np.load(OUT / "neurons.npz")
    st = lambda reg: (sum(NZ[f"sum_{reg}_{h}"] for h in (0, 1)) / sum(int(NZ[f"cnt_{reg}_{h}"]) for h in (0, 1)),
                      sum(NZ[f"sq_{reg}_{h}"] for h in (0, 1)) / sum(int(NZ[f"cnt_{reg}_{h}"]) for h in (0, 1)))
    (ml, ql), (mp, qp) = st("loop"), st("pre")
    loop_d = (ml - mp) / np.sqrt((np.maximum(ql - ml ** 2, 1e-12) + np.maximum(qp - mp ** 2, 1e-12)) / 2)

    rows = {r["id"]: r for r in json.loads((ROOT / "phase3c" / "p3c_c3_original.json").read_text(encoding="utf-8"))["rows"]}
    stuck = np.array([rows[f"m{i}"]["finish"] == "length" and not rows[f"m{i}"]["closed"] for i in range(500)])
    clean = np.array([rows[f"m{i}"]["finish"] != "length" and rows[f"m{i}"]["closed"] for i in range(500)])
    q4 = dict(n_stuck=int(stuck.sum()), n_clean=int(clean.sum()),
              auc_level_alone=r3(auc(level[stuck].astype(float) + np.random.default_rng(0).uniform(-.1, .1, stuck.sum()),
                                     level[clean].astype(float) + np.random.default_rng(1).uniform(-.1, .1, clean.sum()))),
              by_layer=[])
    for l in range(25):
        x = Z["last"][:, l]
        dd = diff_dir[("last", l)]
        # logistic-free outcome direction: stuck-minus-clean mean, fit on half 0
        od = unit(x[fit & stuck].mean(0) - x[fit & clean].mean(0)) if l else dd
        q4["by_layer"].append(dict(
            layer_out=l - 1,
            cos_loop_vs_difficulty_last=r3(loop_dir[l] @ dd),
            cos_loop_vs_difficulty_mean=r3(loop_dir[l] @ diff_dir[("mean", l)]),
            auc_stuck_from_difficulty_dir=r3(auc(x[test & stuck] @ dd, x[test & clean] @ dd)),
            auc_stuck_from_loop_dir=r3(auc(x[test & stuck] @ loop_dir[l], x[test & clean] @ loop_dir[l])),
            auc_stuck_from_outcome_dir=r3(auc(x[test & stuck] @ od, x[test & clean] @ od)),
            neuron_top64_overlap=(int(len(set(np.argsort(-np.abs(loop_d[l - 1]))[:64].tolist())
                                          & set(np.argsort(-E[l - 1])[:64].tolist()))) if l else None)))
    res["Q4_loop_relation"] = q4

    # ---------- POST-HOC control (not in the pre-probe plan): is it just prompt length? ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / "phase3" / "models" / "stock")
    qs = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))
    n_tok = np.array([len(tok(p["q"]).input_ids) for p in qs], float)
    A = np.stack([np.ones(500), n_tok, np.log(n_tok)], 1)
    ctl = dict(auc_length_hard_vs_easy=r3(auc(n_tok[test & (level >= 4)], n_tok[test & (level <= 2)])),
               auc_length_stuck_vs_clean=r3(auc(n_tok[test & stuck], n_tok[test & clean])), by_layer=[])
    for l in range(1, 25):
        x = Z["last"][:, l]
        x = x - A @ np.linalg.lstsq(A[fit], x[fit], rcond=None)[0]  # length (linear + log) regressed out
        d = unit(x[fit & (level >= 4)].mean(0) - x[fit & (level <= 2)].mean(0))
        z = (x - x[fit].mean(0)) / (x[fit].std(0) + 1e-6)
        cent = np.stack([unit(z[fit & (subj == s)].mean(0)) for s in range(len(subjects))])
        pred = (z[test] / np.linalg.norm(z[test], axis=1, keepdims=True) @ cent.T).argmax(1)
        ctl["by_layer"].append(dict(layer_out=l - 1,
                                    difficulty_auc=r3(auc(x[test & (level >= 4)] @ d, x[test & (level <= 2)] @ d)),
                                    stuck_auc=r3(auc(x[test & stuck] @ d, x[test & clean] @ d)),
                                    subject_accuracy=r3((pred == subj[test]).mean())))
    res["posthoc_length_control"] = ctl
    print("POST-HOC length control:", {k: v for k, v in ctl.items() if k != "by_layer"})
    for k in ("difficulty_auc", "stuck_auc", "subject_accuracy"):
        print(f"   length-regressed {k}:", [r[k] for r in ctl["by_layer"]])
    (HERE / "atlas_questions.json").write_text(json.dumps(res, indent=1))

    # ---------- plots ----------
    lay = list(range(-1, 24))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for ro in ("last", "mean"):
        ax[0].plot(lay, list(q1[ro]["accuracy_by_layer_out"].values()), label=f"subject accuracy ({ro} token)")
        ax[0].plot(lay, list(q2[ro]["auc_by_layer_out"].values()), ls="--", label=f"hard-vs-easy AUC ({ro})")
    ax[0].axhline(res["majority_baseline"], ls=":", c="gray", label="subject chance (majority)")
    ax[0].set(xlabel="residual after layer", ylabel="held-out score", title="Q1/Q2: question type is readable from activations")
    ax[0].legend(fontsize=7)
    bl = q1["last"]["best_layer_out"] + 1
    x = Z["last"][:, bl]; x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x[fit], full_matrices=False)
    pc = x @ vt[:2].T
    for k, s in enumerate(subjects):
        ax[1].scatter(pc[subj == k, 0], pc[subj == k, 1], s=9, label=s)
    ax[1].set(title=f"Questions at the last prompt token, after L{bl - 1} (top 2 PCs)", xlabel="PC1", ylabel="PC2")
    ax[1].legend(fontsize=7)
    sc = ax[2].scatter(pc[:, 0], pc[:, 1], s=9, c=level, cmap="viridis")
    ax[2].set(title="Same map, coloured by difficulty level", xlabel="PC1", ylabel="PC2"); fig.colorbar(sc, ax=ax[2])
    fig.tight_layout(); fig.savefig(HERE / "fig_q1_decoding.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(range(24), [r["n_eta2_gt_0p3"] for r in res["Q3_neurons"]])
    ax[0].set(xlabel="layer", ylabel="neurons with eta² > 0.3", title="Q3: subject-selective MLP neurons per layer")
    Lb = int(np.argmax([r["n_eta2_gt_0p3"] for r in res["Q3_neurons"]]))
    top = np.argsort(-E[Lb])[:40]
    m = np.stack([N[subj == k, Lb][:, top].mean(0) for k in range(len(subjects))])
    m = (m - m.mean(0)) / (m.std(0) + 1e-9)
    order = np.argsort(m.argmax(0))
    im = ax[1].imshow(m[:, order], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax[1].set(yticks=range(len(subjects)), yticklabels=subjects, xlabel=f"top 40 subject-selective neurons in L{Lb}",
              title="Which subject lights up which neuron (z-scored mean activation)")
    fig.colorbar(im, ax=ax[1]); fig.tight_layout(); fig.savefig(HERE / "fig_q2_neurons.png", dpi=130); plt.close(fig)

    # ---------- console ----------
    print("majority baseline", res["majority_baseline"])
    for ro in ("last", "mean"):
        print(f"Q1 {ro}:", {k: v for k, v in q1[ro].items() if k != "accuracy_by_layer_out"})
        print(f"   by layer:", list(q1[ro]["accuracy_by_layer_out"].values()))
        print(f"Q2 {ro}: best", q2[ro]["best_layer_out"], q2[ro]["best_auc"], "| by layer:", list(q2[ro]["auc_by_layer_out"].values()))
    print("Q3 per layer n(eta2>0.3):", [r["n_eta2_gt_0p3"] for r in res["Q3_neurons"]])
    print("Q3 split-half r:", [r["split_half_r"] for r in res["Q3_neurons"]])
    print("Q4 stuck/clean", q4["n_stuck"], q4["n_clean"], "AUC level alone", q4["auc_level_alone"])
    for k in ("cos_loop_vs_difficulty_last", "cos_loop_vs_difficulty_mean", "auc_stuck_from_difficulty_dir",
              "auc_stuck_from_loop_dir", "auc_stuck_from_outcome_dir", "neuron_top64_overlap"):
        print(f"   {k}:", [r[k] for r in q4["by_layer"]])


if __name__ == "__main__":
    main()
