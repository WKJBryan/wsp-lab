# Pre-registered protocol — Phase 3b: corrected decoding cross-check

Committed BEFORE either arm runs. Fixes ISSUES #8: Phase 3a used greedy decoding (prohibited
by the model card) with a 6,144-token cap and a score-inflating extraction fallback.

## Purpose

Fairly test whether the original WSP post's published 2-tensor edit improves Qwen3.5-0.8B on
MATH-500 **under the original's own decoding protocol**, on local hardware. This is the
correction of Phase 3a's C3 arm; Phase 3a's blind-editor session is unaffected (its NULL
verdict stands — the editor failure was real in every regime).

## Settings (matching the original's protocol; deviations disclosed)

- Thinking mode ON (`enable_thinking=True`).
- Sampling: temperature 0.6, top-p 0.95, top-k 20, min_p 0 — per the original blog.
- **Presence penalty 1.5**, implemented exactly (OpenAI/vLLM semantics: subtract 1.5 from the
  logit of every token already present in prompt+generation; custom processor — transformers
  lacks it natively). This is the Qwen-recommended anti-loop mitigation.
- repetition_penalty 1.0 (off), seed fixed per arm (20261105).
- **Deviation (budget-controlled, applied to BOTH arms): cap 16,384 tokens instead of 32,768.**
  Reason: single RTX 5090 wall-time; the original's baseline hit 32K on 93% of generations and
  we cannot afford 2 × ~8M+ worst-case tokens twice. Cap-hit rates are reported so the
  truncation effect is visible, never laundered into scores.
- 1 sample per problem, all 500 MATH-500 problems, batch 12 (throughput-only; batch 32 OOM'd
  on a 32 GB 5090 at the 16,384-token KV cache — batch size has no scoring effect).
- **Scoring: last `\boxed{}` only, checked by math-verify (symbolic equivalence). No fallback
  extraction. Truncated generation with no boxed answer = incorrect.** (Phase 3a's
  last-number fallback inflated looping arms — arXiv:2605.07686.)

## Arms

1. **C1-stock** — unmodified `Qwen/Qwen3.5-0.8B`.
2. **C3-original-edit** — the original's published edit applied as released (layers 15/19
   down_proj; built in Phase 3a as `phase3/models/c3_original_edit`).

(Phase 3a's editor candidate `ed_v1` is excluded: it was characterized as a no-op in 3a, and
under corrected decoding a ×1.10 single-row rescale has no mechanism to become active.)

## Pre-registered endpoints

- **TRANSFERS:** C3 ≥ stock + **10.0 points**.
- **PARTIAL:** +5.0 to +10.0.
- **NULL:** < +5.0.
- Secondary (reported, not gated): cap-hit rate, mean/median tokens, per-arm binomial 95% CI
  (±~4.4 pp at these n's).
- Context anchors from the week's independent work (not gates): stock 7.8% @32K/93% cap-hit
  (Carlos Toledo); published edit 50.6% @32K under think35 protocol (openwsp); stock 43–52%
  under non-looping settings. At 16,384 we expect stock lower than 7.8 and C3 lower than 50.6;
  the *gap* is the quantity of interest.
