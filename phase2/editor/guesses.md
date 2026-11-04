# Phase 2 editor guesses (committed BEFORE reveal)

Analysis: `phase2/editor/analyze.py` — weights only, synthetic probes, zero env/data access.

| Net | Guess | Confidence | Key evidence |
|-----|-------|------------|--------------|
| A | `blobs` | high | features 2,3 used (sens 1.00/0.78), 0,1 dead (0.05/0.05); r²=0.975 (additive) |
| B | `cartpole` | high | all 4 features live in θ/ω-dominant pattern (1.0/0.7 vs 0.33/0.28 — pole controller shape); 79 dead layer-2 units (REINFORCE signature; supervised nets ≤10); r²=0.964 (switching hyperplane) |
| C | `parity` | high | all features uniform; r²=0.014 (maximally non-additive); flips/unit 0.98 (roughest); largest weight drift (‖W2‖=37, ‖W4‖=11.7 — overtrained to zero loss on 16 points) |
| D | `random` | high | all features uniform; r²=0.015 (non-additive junk); flips 0.50 (noise level); norms close to fresh init (nothing learnable) |
| E | `interact` | high | features 0,1 used (sens 0.99/1.00), 2,3 dead (0.06/0.07); r²=0.008 (non-additive — quadrant structure) |

## Feature identification (P2)

- Net guessed `blobs` (A): informative features **{2, 3}**
- Net guessed `interact` (E): informative features **{0, 1}**

## Method note

Discriminators used: layer-0 column norms & perturbation sensitivity per input feature; closed-form
linear (additivity) fit of the decision function on N(0,1) and on wide-range inputs; sign-flip
density along random lines (roughness); dead-unit counts; weight-norm drift vs a fresh init.
Machine-readable version: `guesses.json`.
