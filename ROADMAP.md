# Roadmap — portable weight-space medicine

The north-star goal (set 2026-11-17, user-directed): determine whether weight-space
diagnoses are **portable across architectures** — whether the functional map of one
model (which numbers do what) numerically corresponds to another's, so that a
diagnosis proven on model A can be applied to model B without search.

Three falsifiable rungs. Every outcome — including negatives — is a result.

## Phase 11 — wound repair (in flight)

Blind gradient-free repair of the expert-pruned Gemma-4 wound.
- State: baselines complete (`2cc0af5`) — full model 53.1% / 1.6% cap-hit vs wound
  1.6% / 98.4% cap-hit. Editor session pending.
- Verdict rule (pre-registered in `phase11/protocol.md`): SUCCESS = A2 − A1 ≥ +5.0
  AND A2 − A2r ≥ +2.5 on final MATH-500.

## Phase 12 — cross-architecture atlas (pre-registered below)

Map the same functions in both models and test numerical correspondence.
- Qwen3.5-0.8B side: consolidate phase-7 atlas (loop direction → MLP columns) and
  phases-4/10 landmarks (termination row 248069 family).
- Gemma-4 MoE side: same probes on full + wound; expert impact table already
  measured (`phase11/calibration_impact.json`, 30×128).
- Correspondence tests (pre-registered):
  - C1: loop-direction similarity across models (cosine / CKA on shared probe set),
    significance by permutation test (≥95th percentile of shuffled null).
  - C2: termination-circuit signature match — statistical fingerprint (row norm,
    activation statistics, positional behavior) of Qwen's `</think` row vs the
    best-matching Gemma row(s); report rank of the true analog among all rows.
  - C3: cluster-structure comparison — do the 8 kept experts' impact profiles
    cluster like Qwen's MLP column clusters (adjusted Rand index vs matched-size
    null).
- Verdict: CORRESPONDENCE if ≥2 of C1–C3 pass their bars; NO-CORRESPONDENCE
  otherwise (both publishable).

## Phase 13 — the transfer test

Apply Qwen's proven diagnosis to the Gemma wound with ZERO search (and vice versa
where an edit class exists).
- Edit classes eligible: termination-row scaling; loop-direction subtraction from
  analogous columns; router-row rescale (MoE-only).
- Verdict (pre-registered): TRANSFER-POSITIVE if the transferred edit beats the
  matched-norm random control by ≥ +2.5 pts on practice MATH-500 without any
  parameter search on the target model; else TRANSFER-NEGATIVE.
- TRANSFER-POSITIVE is the headline result: portable weight-space medicine.

## Honest framing

The realistic claim spectrum: strong paper (any outcome) → field-defining first
(transfer-positive). Framing in any writeup rides on the data, nothing more.
