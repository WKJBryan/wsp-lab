# Pre-registered protocol — Phase 12: the cross-architecture atlas

> Note added 2026-11-24 (ISSUES #19): the figures quoted below were recorded with the
> defective answer checker and are higher under the corrected one (see ISSUES #19); the
> rules and thresholds in this file are unchanged. (The one accuracy quoted below, 1.6 -> 1.6
> as first reported, is 1.6 -> 3.1 under the final checker.) The verdict drawn from these
> tests was withdrawn on 2026-11-24 (ISSUES #21); the tests below are unchanged as the
> registered text.

Committed BEFORE any Phase-12 probe runs (2026-11-17). The user's question, made
falsifiable: **do Qwen3.5-0.8B (dense, 24 layers, hidden 1024) and Gemma-4 26B-A4B
(MoE, 30 layers, hidden 2816) write the same functions with corresponding number
signatures?** Same symptom already measured on both: repetition loops + a broken
termination mechanism, both responsive to embedding-row boosts.

## Inputs (all already committed or on disk)

- Qwen side: `phase7/ATLAS.md` + `atlas_activation.json` + `atlas_geometry.json`
  (loop direction d, cos(d,u15)=0.71, AUC 0.97–0.99; per-layer down_proj alignment
  profile), phases-4/5/10 dose tables (row 248069 ×1.0–1.5).
- Gemma side: `phase11/a1_wound_final_texts.json` (looping generations),
  `phase11/a0_full_final_texts.json` (terminating generations — the "clean" set),
  `phase11/editor/` pass-2 (embedding boosts: cap-hit 98.4→42) + its exact row ids,
  `phase11/calibration_impact.json` (expert impact table).
- Shared probe prompts: the first 64 MATH-500 problems (both models' texts come
  from the same problems).

## C1 — loop-direction dynamics correspondence

Per model, define the activation-side loop direction d_M = mean(loop states) −
mean(pre-onset states) per layer (Qwen: phase-7 sets; Gemma: built the same way —
loop windows from wound generations, pre windows from the same generations before
onset, clean windows from full-model generations; fit on half, score on half).
Internal validity bar (both models): held-out AUC(loop vs pre) ≥ 0.75.

Cross-model test (tokenizers differ, so token positions cannot be aligned — compare
dynamics, not vectors): for each model, the z-scored projection onto d_M binned by
relative position to loop onset (−500…+500 tokens, 25 bins). **Pass = Pearson r of
the two 25-bin trajectories ≥ 0.5, permutation p < 0.05 (1,000 circular shifts).**

## C2 — termination-circuit signature match

Qwen anchor: embedding row 248069 (`</think>`). Gemma candidates: eos-family rows
(generation_config ids 1, 106, 50) + the editor pass-2 boosted rows (exact ids from
`phase11/editor/apply_edit.py`).
- Behavioral fingerprint (both measured under their own harness): does a row boost
  recover termination without accuracy loss? Qwen: ×1.5 dose (phase 4/5 tables).
  Gemma: pass-2 (cap-hit 98.4→42, accuracy 1.6→1.6).
- Statistical fingerprint: row-norm percentile among all embedding rows, per model.
**Pass = (a) directional behavioral match: termination improves ≥25% relative in
both with accuracy change ≤ −1.0 pt in neither, AND (b) both anchors sit in the
same norm-percentile tercile OR both are outside the median band by the same sign.**

## C3 — writer-geometry (diffuse vs focal) correspondence

Qwen: per-layer max/mean |alignment| onto d (atlas_geometry.json; known diffuse).
Gemma: same statistic computed onto d_Gemma across per-layer writers — dense mlp
down_proj columns AND the 8 surviving experts' down_proj slices.
**Pass = both models diffuse (per-layer max/mean < 3.0 in ≥ 80% of layers) with
overlapping ratio distributions (two-sample KS, p > 0.05), OR both focal with
overlap; FAIL if one diffuse and one focal.**

## Verdict (pre-registered)

**CORRESPONDENCE** if ≥ 2 of C1–C3 pass. NO-CORRESPONDENCE otherwise. Both
outcomes feed Phase 13 unchanged. No metric in this file may be redefined after
the first probe result is seen; problems get logged as ISSUES, not patched silently.

## Constraints

Forward passes and weight-file arithmetic only (int8 for Gemma forwards — same
disclosure as phase-11 calibration; bf16 weight math). No generation beyond what
the saved texts already contain (new forwards are teacher-forced over saved texts).
Blinding is not required — this is an atlas, not a repair claim. Runtime budget:
free analysis only; no eval passes charged.
