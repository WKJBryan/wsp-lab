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
