# Phase 2 results — blind weights-only identification

**Date:** 2026-11-04 · **Protocol:** `phase2/protocol.md` (committed `37f8cec`, before the pool existed)
**Integrity ordering:** guesses committed `f2dfc2b` → sealed manifest committed afterward.

## Verdict (pre-registered endpoints)

| Endpoint | Bar | Result | Call |
|---|---|---|---|
| P1 assignment | PASS ≥4/5, STRONG 5/5 (chance ≈0.8% for full, ≈9.2% for ≥3) | **3/5** | ❌ **FAIL** |
| P2 feature identification | PASS 2/2 exact subsets (chance 1/6 each) | **2/2 EXACT** | ✅ **PASS** |

Per-net: A=`blobs` ✓ · B=`cartpole` ✓ · C=`parity` ✗ (truth `random`) · D=`random` ✗ (truth `parity`) · E=`interact` ✓. Informative-feature subsets: blobs {2,3} exact, interact {0,1} exact.

## What worked

The three confidently-correct identifications rested on the strongest signals — feature usage (layer-0 column norms + perturbation sensitivity) and additivity (closed-form linear fit): the linear 2-feature classifier, the θ/ω-dominant REINFORCE controller (with its 79 dead layer-2 units), and the non-additive 2-feature interaction net. Naming *which inputs matter* — arguably the operationally useful part of "reading" a network — was exact on both scored nets.

## What failed, and why (post-hoc diagnosis)

The parity↔random swap inverted the roughness/norm heuristic:

- I read C (huge weight drift ‖W2‖=37, ‖W4‖=11.7; roughest function, 0.98 flips/unit) as "parity overtrained to zero loss." Truth: C is the **random-labels net**, which memorized **86.5%** of its 4,096 random training labels (final CE 0.348) — memorizing noise is what forces extreme weights and a maximally jagged function.
- I read D (modest norms, moderate roughness 0.50) as "random, couldn't learn." Truth: D is **parity** — 16 discrete points, memorized cheaply; sampled in continuous input space the fitted function is *smoother* than a noise fit, despite parity being maximally non-additive as a Boolean function.

Lesson, stated plainly: weight-space signatures distinguish **structure** (which features, additive or not) reliably, but the roughness/norm heuristics for **what** was fit invert in the memorization regime. An editor with even a handful of labeled probes would catch this trivially (parity evaluates perfectly on the 16 hypercube corners); Phase 2 deliberately allowed zero data access, which is stricter than the original's experiments.

## Honest notes

1. Same agent built the pool and guessed (disclosed from the start); the sealed-manifest + commit-ordering procedure makes the guesses tamper-evident, not the pool construction. The shuffle came from OS randomness; the informative-feature draws were unseeded (`secrets`).
2. The "random" task memorized more than intended — a 17K-parameter net with 2,000 Adam steps absorbs most of 4K random labels. That made it a *harder*, not easier, distractor.
3. P1 chance levels quoted for a uniformly random permutation of five items.

## Takeaway for the WSP question

Reading weights alone is enough to extract a network's wiring facts (which inputs matter, linear vs interactive) without any data — but task identity in the memorization regime fooled this editor. The failure mode is informative: it is exactly where a small evaluation budget (the Phase-1 setting) would have rescued the diagnosis.
