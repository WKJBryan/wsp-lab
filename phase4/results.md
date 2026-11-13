# Phase 4 results — external editor models, blind, on the working stack

**Date:** 2026-11-12/13 · **Protocol:** `phase4/protocol.md` (committed `a53977a`, before either session)
**Engine:** vLLM 0.33 under WSL2 (phase-3c stack) · **Editors:** host-configured models via dynamic-workflow
subagents, per the user's instruction ("spawn subagent to be able to use it").

## The question

Can a different, small editor model — blind, no hints — diagnose and repair the loop pathology
in 10 budgeted passes, where the original's big hinted editor succeeded and our agent-editor
(phase 3a, broken stack) achieved discovery but not repair?

## Final numbers (identical harness, engine, decoding, scorer as phase 3c)

| Arm | MATH-500 | Answered | Terminated (closed) | Mean tokens |
|---|---|---|---|---|
| Stock (phase-3c baseline, carried) | **6.0%** | 9.8% | ~9% | 30,108 |
| **GLM-5.3-Flash, blind (6/10 passes)** | **25.2%** | 98.0% | **76%** | **10,712** |
| C2: matched-norm random noise, same rows | **2.2%** | 4.2% | ~4% | 31,639 |
| mimo-v2.6-flash, blind (1/10 passes) | — no edit produced — | | | |
| Original's published edit (their editor, hinted) | 39.4% | 83.2% | 74% | 20,645 |

**Pre-registered verdict, GLM-5.3-Flash arm: SUCCESS** (25.2% ≥ 16.0% bar; +19.2 points, 4.2×
stock). The C2 control shows the result is the *content* of the edit, not its magnitude:
random noise of identical norm on the identical rows scores 2.2% (stock-like, if anything
slightly harmful).

## Arm 1 — GLM-5.3-Flash: blind success, fully verified

- **Diagnosis (its own, from evidence):** thinking never terminates because the exit token
  `</think>` always loses by 8–13 nats to peaked top-1 logits; plus a scorer-format interaction
  (comma-grouped numbers). Both established by offline logit probes before any edit.
- **Edit (byte-verified by the harness):** ONE tensor of 488, exactly 3 rows of the tied
  embedding — `</think>` (row 248069) ×1.5503, comma (row 11) ×0.5000, `<|im_end|>`
  (row 248046) ×1.1500 — pure rescalings (cosine = 1.0), 3,072 of 873M values (0.0004%).
- **Session record:** 6 passes, all pre-logged with hypotheses (`session_glmflash/ledger.md`),
  including one tested-and-reverted regression (`im_end` ×1.45 → 42.2% practice) and a second-seed
  re-verification. Practice set: 12.5% → 54.7%/50.0% (two seeds); termination 18.8% → 100%;
  mean tokens 7,253 → ~934. On MATH-500 the gain transfers at 25.2% (practice→final drop expected:
  harder problems, and the comma fix is practice-format-specific).
- **Disclosed loophole (ISSUES #12):** the editor tuned scales via unlimited OFFLINE transformers
  generation on the practice problems — legal under a budget that gated only `eval_pass.py`, but
  it makes the pass budget partially decorative. Kept identical for arm 2. Lesson for future
  protocols: budget every evaluation channel (the original's kernel could not run the model at all).

## Arm 2 — mimo-v2.6-flash: competent inspection, no operation

Three session segments (initial ask + two explicit continuations; final one errored on a
missing structured report). Ground truth from the sandbox: **1 pass spent (baseline:
12.5% / 18.8% closed / 82.8% cap-hit — byte-identical numbers to GLM-Flash's pass 1,
confirming harness determinism across editors), zero weight edits, model md5-identical to
pristine.** Its genuine contributions: independently found and minimally fixed the evaluator's
duplicate-keyword bug (as GLM-Flash had — the bug shipped twice; see ISSUES #11), verified
budget-integrity and weight-integrity by checksum, read the architecture correctly, and formed
a reasonable but untested hypothesis (reduce overconfident self-repetition via late-layer
norm/embedding-scale perturbation).

**Verdict: no edit attempted — NULL by non-attempt.** The finding is about long-horizon agency:
GLM-5.3-Flash executed the entire multi-hour session inside one instruction; mimo-v2.6-flash
returned interim statuses and never entered the edit loop even under explicit continuation.

## The harness-bug saga (ISSUES #11, in full honesty)

`eval_pass.py` shipped a duplicate `closed_rate` keyword (introduced in the accuracy edit,
`1d856f7`). Both editors hit it and both fixed it minimally, preserving originals and budget
integrity — two independent, correct catches. The author (this agent) then failed to fix the
repo copy TWICE, in the second attempt relying on `ast.parse`, which accepts repeated keyword
arguments; `py_compile` is the check that decides. The repo file is fixed and compiler-verified
(`fb28216`). Meta-lesson symmetric with the whole project: verify with the tool that actually
decides, not the one that looks like it should.

## What this phase adds to the WSP picture

1. **Blind small-model weight surgery is real but not uniform across editors**: 1 of 2
   flash-tier editors achieved a pre-registered SUCCESS (6.0 → 25.2) blind; the other inspected
   competently and stalled. Cross-editor capability variance is now measured, not assumed.
2. The blind gap to the hinted original narrows dramatically: hinted gpt-6-astra (50 passes,
   their stack) 7.9 → 48.1; blind GLM-5.3-Flash (6 passes, our stack) 6.0 → 25.2.
3. Cheap edits, expensive verification: both successful arms (the original's and ours) rest on
   a few hundred to a few thousand values; everything else is measurement discipline.

## Files

Sessions: `phase4/session_glmflash/`, `phase4/session_mimo/` · Evaluator: `phase4/eval_pass.py`
· Finals: `phase3c/p4_glmflash*.json`, `phase3c/p4_c2_random*.json` (generations gzipped).
