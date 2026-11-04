# Pre-registered protocol — Phase 3: blind (no-hint) Qwen3.5-0.8B repair

Committed BEFORE any model download or evaluation. This is the condition the original's logs never ran at full budget: improve the model with **no failure-mode information of any kind** in the editor's instructions.

## Research question

Can a different agentic editor (ZCode / GLM-5.3), given only a stock Qwen3.5-0.8B's weight files, a code sandbox, and a strict evaluation budget — and told NOTHING about known failure modes — improve the model's MATH-500 accuracy by direct weight edits, with no gradient training?

## Setup

- **Target:** stock Qwen3.5-0.8B instruct model (exact HF repo id resolved at download and recorded in `phase3/results.md`; local download, bf16).
- **Editor inputs:** the model's safetensors files, tokenizer, the eval harness, and this instruction only:
  > "Improve this model's MATH-500 accuracy. You may edit weights directly. No gradient training. You have 10 evaluation passes."
- **Practice set (budget passes):** 128 fixed problems from the GSM8K train split (ids fixed by seed before editing), generation cap 4096 tokens.
- **Final benchmark:** full MATH-500 (500 problems), greedy decoding, cap 6144 tokens, single pass, `\boxed{}` extraction with light normalization. Identical settings for every arm.
- **Pre-registered fallback:** if any final arm exceeds 2 h wall-clock, all arms switch to the first 250 problems (equally applied) and results are labeled as such.

## Editor rules

1. Budget: **10 evaluation passes** (each = one practice-set run). Weight statistics offline are unlimited. Any editor-driven diagnostic generation counts as a pass.
2. Edits are direct tensor arithmetic on safetensors. No backprop, no optimizers, no training, no prompt changes, no sampler changes.
3. Every pass is logged in `phase3/editor/ledger.md` BEFORE it runs: hypothesis → planned edit → evidence cited.
4. **Contamination disclosure (cannot be avoided, so it is declared):** the editor researched the original WSP post and knows its public findings, including the existence of a repetition/loop failure mode. Mitigation, pre-registered: any edit that targets looping/repetition MUST cite concrete observed evidence from this model's practice-set generations (passes 1–2) — file, problem ids, quoted text — before the edit is applied. Edits cannot be justified by "the original did this."
5. The editor may not consult the original's released logs or published edited model during the editing session.

## Pre-registered endpoints (final MATH-500, greedy, identical settings)

- Baseline = stock model measured under this exact protocol (C1).
- **SUCCESS:** edited accuracy ≥ baseline + **10.0 points** AND ≤ 1% of weights touched.
- **PARTIAL:** +5.0 to +10.0 points (with ≤1% weights).
- **NULL:** < +5.0 points, or >1% of weights touched.
- Loop-rate (fraction of final generations hitting the token cap) reported for every arm.

## Addendum (2026-11-04, before any arm comparison — harness fixes, ISSUES #6)

The first stock baseline run revealed two harness-side issues, fixed identically for every arm
before any comparisons were made:

1. The tokenizer's chat template **defaults thinking off** (renders an empty `<think>` block).
   All arms now generate with `enable_thinking=True` — the model's advertised mode and the
   original's setting. The void no-thinking baseline (0/128 boxed-extraction, 25% cap-hits)
   is kept on disk for the record.
2. Answer extraction extended: `\boxed{}` first, then last-number fallback (this model answers
   GSM8K in prose without boxes). Applied to all arms; saved generations re-scored, not re-run.

## Controls

- **C1 untouched stock model** (the baseline itself).
- **C2 random-edit control:** Gaussian perturbation with Frobenius norm matched to the editor's final edit, applied to the same tensors the editor touched; one trial; final protocol.
- **C3 original's-published-edit cross-check:** compute the diff between the stock model and the original's released edited model (RikiyaT/Qwen3.5-0.8B-weight-space-programming), apply that diff to our stock copy, evaluate under OUR protocol. This tests whether their published edit transfers under different decoding settings (greedy vs. their sampled) and a different harness — an independent cross-check of the original's headline result, which no one has published under altered settings.
