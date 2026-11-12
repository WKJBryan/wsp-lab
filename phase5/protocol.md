# Pre-registered protocol — Phase 5: optimization beyond the original's edit

Committed BEFORE any analysis or edits. This is an optimization phase, not a blind
replication: the editor (this agent) knows all prior results. Disclosed up front.

## Research question

Starting from stock + the original's published edit (phase-3c edited model: 39.4% single-sample,
26.4% cap-hit), can INFORMED weight-space edits — designed from measured geometry of the
existing edit, not blind search — reduce the stuck rate further and raise MATH-500 accuracy?

## Starting model and baseline

- Start: `phase3/models/c3_original_edit` (stock + their released 2-tensor edit, verbatim).
- **Baseline for this phase = the starting model re-measured under the final protocol below**
  (not carried over from the 39.4% single-sample number — sampling design differs).

## Final protocol (both arms identical)

- Full 500 MATH-500 problems, thinking on, temp 0.6 / top-p 0.95 / top-k 20 / presence
  penalty 1.5, cap 32,768, **n=4 samples per problem** (matching the original's averaging;
  score = mean accuracy across the 4 samples). vLLM engine, `phase3c/rescore.py` extraction.
- Stuck rate (cap-hit across all 2,000 generations) reported as co-primary.

## Exploration budget

- **12 practice passes** max: 64 fixed GSM8K-train problems (phase-4 practice set), cap 8192,
  same decoding, n=1. Each pass returns accuracy + termination stats + sample tails.
- Every pass pre-logged in `phase5/editor/ledger.md` with hypothesis → edit → result.
- Weight analysis (CPU on safetensors, reading saved generations) unlimited and free.

## Edit rules

- Direct weight arithmetic only — no gradients, no optimizers, no training (the WSP frame).
- Edits must be motivated by measured evidence (edit anatomy, layer census, or practice-pass
  diagnostics), logged before evaluation.

## Pre-registered thresholds (final, n=4 protocol)

- **SUCCESS:** candidate ≥ starting-model baseline **+5.0 points** AND stuck rate ≤ baseline's.
- **PARTIAL:** +2.5 to +5.0, or ≥ +5.0 with a stuck-rate regression ≤ +3 points (trade-off
  must be visible, not hidden).
- **NULL:** below that, or no candidate produced.

## Planned ladder (cheapest first; later steps only if earlier ones justify)

1. **Anatomy** (free): which of the 3,584 input columns their edit really changes; the edit's
   dominant direction(s) and dose per layer; whether L15/L19 structures align; re-serialization
   noise floor vs true edit magnitude.
2. **Dose-response:** their delta scaled ×1.25 and ×1.5 (two passes). If the original dose is
   sub-optimal, this shows the curve's shape immediately.
3. **Layer extension:** apply the measured geometry to the closest-matching untreated layers
   (ranked by analysis), scaled conservatively (×0.5 of the L15/L19 dose initially).
4. **Residual targeting:** analyze still-stuck practice generations for a distinct second loop
   mode; only pursue if clearly present.

## Controls

- C1 = starting model under the final protocol (the baseline itself).
- If a candidate wins: report its per-tensor diff norm vs the starting model (edit size).
- No random-edit control at final scale (GPU cost); the phase-1 lesson about matched-norm
  controls applies to novel edits, and steps 2–3 are perturbations of an edit whose causal
  effect is already established in phase 3c.
