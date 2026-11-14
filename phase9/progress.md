# Phase 9 progress log (per condition; folded into results.md at the end)

## Condition 1 — glm5.3-flash-team — **HURTS: 10.6% vs baseline 25.2%** (2026-11-14)

- Session: 6/10 passes (4 unspent); all pre-logged. Byte-verified edit: UNTIED lm_head +
  row 248069 += 0.135·u (u = measured mean loop-position hidden direction), index/config fixed.
  A genuinely novel edit class — no phase-4 editor did this.
- Deliberation quality was HIGH: red-team caught 4 real errors pre-spend (wrong-stack evidence,
  measured premature-close tail, false norm-percentile claim, index landmine); judge ran its own
  vLLM lift calibration; executor's brackets disciplined, one pass lost to a pipe (honestly
  reported; harness now persists JSON to file — fix applied post-condition, no scoring change).
- Final: 10.6% / 83.2% cap-hit / mean 28,151 tokens. The conservative additive dose that closed
  55% of loops on 64 easy practice problems (cap 8K) fails on hard MATH-500 reasoning (10–30K
  tokens): the multiplicative ×1.5 rescale the red-team REJECTED for measured premature-close
  risk survives finals far better (GLM single: 25.2%, 24% cap-hit).
- Interpretation (1-condition, provisional): deliberation → risk-aversion → under-dosing →
  failure at scale. The practice set (easy, short, cap 8K) is a poor proxy for dose selection —
  practice-level noise (±6–9% at n=64) further misled dose choice (α picked on a single 40.6%).
- Qualitative endpoint: red-team DID predict a real cliff (spam at α≥0.25; executor's 0.225 pass
  showed the signature) — the safety analysis was correct; it was the EFFICACY that the caution
  sacrificed.

## Harness notes

- Post-condition-1 fixes (disclosed, no behavior change): eval_pass.py now tees its JSON to
  `eval_pass_last.json` (three sessions have lost outputs to pipes); stale "accuracy
  intentionally NOT shown" note removed (contradicted its own output; flagged by the observer).

## Condition 2 — mimo-flash-team — **NULL: degenerate session, no edit attempted** (2026-11-14)

- Ground truth: 1/10 passes (baseline only), model md5-pristine, no edit applied.
- Every role returned fragments/leaked chain-of-thought instead of deliverables; the judge's
  "verdict" was internal monologue; the executor reported a garbled spec and executed nothing.
- Cross-format pattern (now 2 data points): mimo-flash succeeds on SHORT single-constraint
  steps (phase 4b: 27.8%) and disintegrates on long multi-role asks (phase 4 arm 1, and here).
  The bottleneck is long-ask discipline, not surgical capability.
- Suspected (unproven, zero experimental impact): executor quoted GLM-phase-4's apply_edit.py
  signature — possible repo read outside sandbox; nothing was applied and no extra pass was
  spent, so no measurement was affected. Sandbox-only rule remains soft-boundary + audit.
- Baseline comparison for this condition is therefore not applicable (no candidate produced).

## Condition 3 — mimo-pro-team — **NULL: degenerate session, no edit** (2026-11-14)

- Ground truth: 1/10 passes (baseline), model pristine, no strays.
- Worse than c2: analysts returned planning monologues; critique truncated; **judge and
  executor returned empty strings** — the same session-end-without-output failure family as
  the phase-4 typed deaths, now in plain-text form.
- Verdict on the mimo team conditions (c2, c3): the multi-role long-ask format is incompatible
  with both mimo tiers in this harness, while identical scripts ran end-to-end on GLM-5.3-Flash.
  Reinforces the phase-4 finding: **session format dominates model tier.** mimo models succeed
  on short self-contained asks (4b/4c) and fail when asks carry long interpolated role context.
- Consequence for remaining conditions: c4 (glm5.3 team) + c5 (glm5.3 single) proceed as
  planned; c6 (mixed) keeps all four models but assigns producer-critical roles (judge,
  executor) to GLM-family models with mimo on analyst/red-team seats (disclosed assignment);
  c7 (interleaved) stays GLM-5.3-Flash.

## Condition 4 — glm5.3-team — **24.4%** (2026-11-14; verdict vs c5 pending)

- Session: superb (2/10 passes — most efficient ever; single edit pass, no bracketing needed).
  Byte-verified: untied lm_head, row 248069 ×1.3516, flags/index fixed.
- Committee SYNTHESIS: A's target+dose on B's untie machinery (red-team falsified B's
  contamination claim at ×1.35 — it held only at ×2; judge merged correctly).
- Practice 12.5→48.4 / closed 18.8→78.1. Final: **24.4%**, cap-hit 62.8%, mean 22,815.
- Note: the derived dose bracket [1.30,1.40] (from measured step-0 spike margins, s_crit 1.535)
  sits BELOW the loners' empirically-safe ×1.5 — theoretical caution cost a few points again,
  though far less than c1's additive-dose collapse. HELPS/NEUTRAL/HURTS awaits c5.

## Condition 5 — glm5.3-single — **23.0%** (2026-11-14) → **c4 verdict: NEUTRAL (+1.4)**

- Session: 7/10 passes; most sophisticated edit of the project (byte-verified: untied head,
  ridge-LS conditional exit-boost ‖Δ‖=0.56 + 5 format-row scalings ×0.1–0.4).
- Final: 23.0%, cap-hit 63.0%, mean 21,977. Practice ~57 halved at finals, as with every arm.
- c4 vs c5: 24.4 vs 23.0 = NEUTRAL (within ±4.4). Prediction on record held.
- Notable: the BIGGEST model is the WEAKEST loner (23.0 vs flash-tier 25.2/27.8/27.8) — its
  conditional-boost machinery out-engineered the crude rescale but did not out-score it.

## Solo blind-surgery leaderboard (all loners, identical rules)

mimo-v2.6-flash 27.8 = mimo-v2.6-pro 27.8 > GLM-5.3-Flash 25.2 > GLM-5.3 23.0 (stock 6.0;
random control 2.2). Tier/family does not buy score — if anything, inverse.
