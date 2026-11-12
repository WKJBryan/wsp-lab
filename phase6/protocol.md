# Pre-registered protocol — Phase 6: systematic hypothesis-driven optimization

An EXPLORATORY campaign (not confirmatory): each experiment is an educated, calculated guess,
logged before running, with a prediction and a verdict after. The log is the deliverable.

## Frame

Starting point: the best model from Phase 5 (dose curve + any extension). Every experiment
follows the same record structure, committed individually:

```
experiments/eNNN_<slug>/
  HYPOTHESIS.md   — the guess, the mechanism, and a numeric PREDICTION
  edit.py         — the exact weight arithmetic (reproducible)
  result.md       — practice-pass numbers, verdict: CONFIRMED / REFUTED / INCONCLUSIVE
```

Master index: `phase6/EXPERIMENTS.md` (one row per experiment, cumulative).

## Budget and decision rules

- **30 practice passes** for the whole campaign (64 problems, cap 8192, phase-5 eval; ~5 min each).
- Practice n=64 has binomial SE ≈ 5–6 points: a single-pass swing < 8 points is treated as
  noise (INCONCLUSIVE) unless confirmed by a second seed. Swings ≥ 8 points act on.
- Finals (at most 2 candidates): 500 MATH-500 × n=4 samples, phase-5 protocol, vs the
  Phase-5 winner as baseline.
- **Collateral-damage gate:** any candidate reaching finals also runs a 30-prompt general
  sanity check (mixed plain QA + instruction-following prompts, scored by inspection rules:
  does it answer, is it coherent, does it terminate). Visible regression = disqualified,
  whatever the math score.

## Hypothesis families (initial ordering by cost × expected value)

1. **Layer extension** — apply the measured direction subtraction to the most v-aligned
   untreated columns of neighboring layers (14, 16, 18, 20), conservative dose (≤0.5× of L15/L19).
2. **Independent layer doses** — L15 and L19 doses tuned separately (their original doses
   differ; maybe one is under-medicated).
3. **Column sparsity** — trim to the 32 strongest columns vs extend to 96 aligned ones
   (is 64 structural or arbitrary?).
4. **Direction refinement** — recompute the loop direction from ACTIVATIONS on stuck
   generations (weight-space v is a proxy; the activation-space truth may differ slightly).
5. **Attention-side damping** — o_proj rows aligned with v, small negative scaling
   (loops are self-reinforcing through attention).
6. **Token-side nudges** — embedding rows of the dominant repeated tokens ("Wait", "So,")
   relative to v.
7. **Whatever Phase 5's dose curve reveals** (if 1.5 wins, dose 2.0 belongs here; if 0.75
   wins, dose 0.5).

New families may be added mid-campaign — logged as new rows here with a dated note.
