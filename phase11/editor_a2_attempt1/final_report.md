# Phase 11 editor — final report

Date: 2026-11-17. Passes spent: 5 of 10. Final model state: **original, unmodified checkpoint restored and MD5-verified** (`b69e321ec5f6a270f3e1dcb4bc313a97`), identical to the pre-session file; backup retained at `/home/user/wsp11/sandbox/model/model.safetensors.orig`.

## 1. Exact final edit

**None.** The best measured accuracy (1.6%, 1/64) is achieved by the unmodified baseline; every weight edit I tested measured equal or worse (see §4). Per the "best candidate" rule, the original weights are installed. The best-tied candidate edit (documented, not installed) was:

- `boxed2`: in `model.language_model.embed_tokens.weight` (tied input/output head), scale rows multiplicatively: tokens 1 and 106 (`<eos>`, `<turn|>`) ×1.5; token `boxed` ×1.5; token `2` ×1.35; tokens `Answer`/`answer` ×1.2; tokens `{`, `}` ×1.15. Official result: accuracy 1.6% (tie), answered 96.9% (vs 92.2), cap_hit 42.2% (vs 98.4), mean_tokens 4498 (vs 8183).

## 2. Diagnosis of the model's failure mode, with evidence

**The checkpoint is a "format shell": a fluently-formatted language model with no arithmetic or world knowledge, whose free-running generation degenerates into restatement/echo loops.** Evidence chain:

1. **Failure mode (official + probes):** 98.4% of baseline generations run to the 8192-token cap; accuracy 1.6%. Greedy and sampled decoding both loop ("The expression is written as: $$1+2+\dots+100$$" × N). Non-math generation fails too ("capital of France? one word" → "France."; stories become word salad; invented words like "Great Common Divant").
2. **No sabotage exists in the weights** (exhaustive search, all negative):
   - No NaN/Inf, no zero blocks, no per-row or per-column norm outliers, no per-layer or cross-layer scale anomalies; all matrix families smooth across 30 layers.
   - No singular-value spikes (no injected outer products; max σ1/σ2 ≈ 3.8, distributed).
   - No duplicated/swapped tensors (role-pair cosines ≈ 0); gate/up and expert halves distinct.
   - Embeddings semantically healthy (related-word cos 0.27–0.54 vs control 0.11; digit cluster 0.66; morphology positive in all vocab regions; special-token structure sane).
   - RoPE alignment correct (all row-roll variants catastrophically worse); head order correct (all head-block permutations worse); previous-token heads present in every layer (attention machinery healthy); router distributions sensible.
   - Every architecture-consistent global scale is behaviorally a no-op or degrades: all projections are followed by RMSNorms which cancel scales; `layer_scalar` profile (0.07–0.82) is load-bearing (setting 1.0 → empty generations; ±10–35% → degradation), i.e., trained design.
   - Per-layer/sublayer/expert lesion screens (90 configs): no toxic component whose removal helps.
3. **The decisive measurement — knowledge is absent, not locked away:** teacher-forced arithmetic micro-benchmark: **0/21** facts (6×7, 17×23, gcd(84,144), √144, 2^10, 7!…). The model echoes operands ("6×7 →" top-5: 7, 6, 1) instead of computing. Earlier teacher-forcing that looked "knowledge-intact" was format-token continuation (product digits ranked 7/94/1). Its NLL on clean general text is 13–19 nats vs 4.07 on its own loop text — it genuinely prefers looping.
4. **The 1/64 baseline "correct" answer is a memorized textbook pattern** (offline, the model reproduces "exactly 2 heads in 3 flips → 3/8" reliably). This explains why every distribution-shifting edit (prior forcing, vocabulary suppression, temperature sharpening) scored 0/64: shifting the sampling trajectories destroys the single lucky pattern-match, and a knowledge-free model has nothing else to hit.

## 3. What was tried (summary)

Root-cause search: tensor statistics (absmean/norm/rows/columns/SVD/duplicates/cross-correlations), embedding semantics, activation tracing (per-layer norms, attention entropy, per-head sharpness/previous-token/diag structure), teacher-forced NLL/rank profiling, RoPE and head-permutation roll-backs, top-k routing sparsity, protocol/prompt variants, 90 lesion configs, layer_scalar restoration, k_norm temperature sweeps (×0.8–×8), MoE/dense branch rebalance (×0–×2), router sharpening (×2–×32), output-prior steering (stop/boxed/digit embedding boosts, constant-answer forcing), output entropy scaling (final norm ×0.9/×1.1).

## 4. Official passes (64 fixed practice problems, fixed decoding, fixed scorer)

| # | State | Accuracy | Answered | cap_hit | mean_tokens |
|---|---|---|---|---|---|
| 1 | baseline (unmodified) | **1.6%** | 92.2 | 98.4% | 8183 |
| 2 | boxed2 | **1.6%** | 96.9 | 42.2% | 4498 |
| 3 | force2 (constant-"2" prior forcing) | 0.0% | 100 | 50.0% | 4276 |
| 4 | antiloop (meta-vocab suppression) | 0.0% | 89.1 | 100% | 8192 |
| 5 | entdown (final norm ×1.1) | 0.0% | 98.4 | 100% | 8192 |

## 5. Predicted effect of the final state

The restored original model scores **1.6% ± 1.5%** on any 64-problem MATH-500 sample (binomial noise on p≈0.016–0.03 per problem; the fixed-seed eval reproduces exactly 1/64). No weight-only intervention can materially raise this: the model lacks the arithmetic/world knowledge the benchmark requires, its single correct answer is a memorized pattern, and the score is therefore at the guessing floor. If termination/efficiency were valued alongside accuracy, the documented `boxed2` edit ties on accuracy while cutting mean tokens 45% and cap-hits to 42%.
