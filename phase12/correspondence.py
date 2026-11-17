"""Phase 12: score the pre-registered correspondence tests C1-C3.

Reads atlas_qwen.json + atlas_gemma.json (probe_atlas.py outputs) and the
committed behavioral measurements; writes correspondence.json and prints the
verdict. Bars are pre-registered in phase12/protocol.md — do not adjust here.

Usage (any python with numpy/scipy):
  python correspondence.py
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
P11 = HERE.parent / "phase11"
P3C = HERE.parent / "phase3c"


def perm_p(r_obs, traj_a, traj_b, n=1000, seed=20261117):
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        k = rng.randrange(1, len(traj_b))
        shifted = traj_b[k:] + traj_b[:k]
        r = float(np.corrcoef(traj_a, shifted)[0, 1])
        if abs(r) >= abs(r_obs):
            hits += 1
    return hits / n


def main():
    q = json.loads((HERE / "atlas_qwen.json").read_text())
    g = json.loads((HERE / "atlas_gemma.json").read_text())
    out = {}

    # ---- C1: loop-direction dynamics ----
    med_auc_q = float(np.median(q["aucs_loop_vs_pre"]))
    med_auc_g = float(np.median(g["aucs_loop_vs_pre"]))
    valid = med_auc_q >= 0.75 and med_auc_g >= 0.75
    r_obs = float(np.corrcoef(q["trajectory"], g["trajectory"])[0, 1])
    p_perm = perm_p(r_obs, q["trajectory"], g["trajectory"])
    c1_pass = valid and r_obs >= 0.5 and p_perm < 0.05
    out["C1"] = dict(median_auc_qwen=round(med_auc_q, 3),
                     median_auc_gemma=round(med_auc_g, 3),
                     internal_validity=valid,
                     trajectory_r=round(r_obs, 3),
                     permutation_p=round(p_perm, 4),
                     passed=bool(c1_pass))

    # ---- C2: termination-row fingerprint ----
    # behavioral: qwen = stock vs original-edit terminations (phase-3c cross-check);
    # gemma = editor pass-1 vs pass-2 cap-hit (phase-11, identical scorer class)
    q_stock = json.loads((P3C / "p3c_stock_rescored.json").read_text())["summary"]
    q_edit = json.loads((P3C / "p3c_c3_original_rescored.json").read_text())["summary"]
    g_base = json.loads((P11 / "ed_pass1.json").read_text())["summary"]
    g_boost = json.loads((P11 / "ed_pass2.json").read_text())["summary"]

    def term_metrics(base_cap, boost_cap, base_acc, boost_acc):
        rel = (base_cap - boost_cap) / base_cap            # cap-hit reduction = termination recovery
        acc_loss = base_acc - boost_acc
        return rel, acc_loss

    q_rel, q_loss = term_metrics(q_stock["cap_hit_rate"], q_edit["cap_hit_rate"],
                                 q_stock["accuracy"], q_edit["accuracy"])
    g_rel, g_loss = term_metrics(g_base["cap_hit_rate"], g_boost["cap_hit_rate"],
                                 g_base["accuracy"], g_boost["accuracy"])
    behav = (q_rel >= 0.25 and g_rel >= 0.25 and q_loss <= 1.0 and g_loss <= 1.0)

    # statistical: same tercile of embedding-row norm percentile
    def tercile(p):
        return 0 if p < 1 / 3 else (1 if p < 2 / 3 else 2)
    q_pct = q["term_row_norm_percentile"][str(q["term_rows"][0])]
    g_pcts = {k: v for k, v in g["term_row_norm_percentile"].items()}
    same_tercile = any(tercile(q_pct) == tercile(v) for v in g_pcts.values())
    c2_pass = behav and same_tercile
    out["C2"] = dict(qwen_term_recovery_rel=round(q_rel, 3), qwen_acc_loss=round(q_loss, 2),
                     gemma_term_recovery_rel=round(g_rel, 3), gemma_acc_loss=round(g_loss, 2),
                     behavioral_match=behav,
                     qwen_row_percentile=q_pct, gemma_row_percentiles=g_pcts,
                     same_tercile=same_tercile, passed=bool(c2_pass))

    # ---- C3: writer geometry (diffuse vs focal) ----
    rq = [q["writer_ratios"][str(i)]["ratio"] for i in range(q["n_layers"])]
    rg = [g["writer_ratios"][str(i)]["ratio"] for i in range(g["n_layers"])]
    q_diffuse = sum(1 for x in rq if x < 3.0) / len(rq) >= 0.80
    g_diffuse = sum(1 for x in rg if x < 3.0) / len(rg) >= 0.80
    ks = sps.ks_2samp(rq, rg)
    structure_agree = (q_diffuse == g_diffuse)
    c3_pass = structure_agree and ks.pvalue > 0.05
    out["C3"] = dict(qwen_ratio_minmax=[min(rq), max(rq)], gemma_ratio_minmax=[min(rg), max(rg)],
                     qwen_diffuse=q_diffuse, gemma_diffuse=g_diffuse,
                     structure_agree=structure_agree, ks_stat=round(ks.statistic, 3),
                     ks_p=round(ks.pvalue, 4), passed=bool(c3_pass))

    passed = sum(1 for c in ("C1", "C2", "C3") if out[c]["passed"])
    out["verdict"] = "CORRESPONDENCE" if passed >= 2 else "NO-CORRESPONDENCE"
    out["tests_passed"] = passed
    (HERE / "correspondence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
