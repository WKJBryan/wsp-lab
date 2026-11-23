# Pre-registered protocol — Phase 14: the blind editor under the original authors' rules

Committed BEFORE any editor session and before any practice pass (2026-11-24). Do not edit after
the session begins; deviations go to ISSUES.md.

## Why this phase exists

The gap between this repository's blind editors and the original authors' editor was reported as
"the hint is worth about 12 points". It does not measure the hint (ISSUES #23). The two setups
differed in four ways at once:

1. The original's task instructions named the failure (repetition inside the thinking); this
   repository's named nothing.
2. The original's rules forbade the quick fix that all ten of this repository's blind sessions
   converged on: edits to the embedding / output layer and the final norm, whole-tensor constant
   scaling, and making the model skip or close its thinking early.
3. The original's editor practised on 7,275 MATH training problems, up to 512 per pass at the
   full 32,768-token cap; this repository's practised on 64 GSM8K problems at an 8,192-token cap.
4. The original's editor was a frontier model; this repository's were small models.

This phase removes differences 2 and 3 for one blind editor and keeps 1 and 4.

Sources for the original's rules: `RikiyaT/lm_understand-logs`, `exp8_math/README.md` and the
first task instructions in `exp8_math/runs/p50c/turns/t001/prompt.txt`.

## Setup

- **Model:** stock Qwen3.5-0.8B, the checkpoint every phase since 3c has used, in a sandbox. The
  editor is not told its name.
- **Editor:** one context-sterile session of the `mimo-pro-editor` agent type. If that model
  cannot run, `glm-flash-editor`; whichever ran is recorded in the results. One autonomous
  session. Its complete task instructions are `EDITOR_PROMPT.md` in this folder: the original's
  section structure and rules, minus the sentences that name the failure.
- **Rules, enforced in software** (`save_edit.py`; a candidate without its receipt is never
  evaluated): only weight values change, with the same tensor names, shapes and dtypes; the final
  norm, the input embedding and the output layer may not be edited; no changed tensor may be only
  a constant multiple of the original; config, tokenizer, chat template and decoding are fixed.
  No gradients, autograd, optimizers or fitting to a loss. Generation on the model only through
  the evaluation tool (enforced by instruction and audited afterwards from the session's
  commands). Weight inspection and teacher-forced forward passes are free.
- **Practice pool:** `practice.json`, 128 problems from the MATH training split (all seven
  subjects; difficulty mix mirroring MATH-500; seed 20261124; no text overlap with the scored
  500), built by `make_practice.py` before this protocol was committed. The editor sees the
  questions and the gold answers.
- **Evaluation tool** (`eval_pass.py`): every pass runs the same 128 problems under the
  measurement protocol below. Budget: 15 passes. A pass is charged before the engine starts and
  refunded if the engine fails; the hypothesis is written to the ledger before the run. The tool
  returns accuracy by the boxed rule, accuracy by the repository's extractor, the share that
  closed its thinking, the share cut at the cap, mean tokens and counts per difficulty level, and
  keeps every text with its extracted answer and grade.
- **Measurement protocol** (the original authors'): the reference benchmark question
  ("Problem: … Please reason step by step, and put your final answer within \boxed{}."), thinking
  enabled, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens,
  one sample per problem, a fixed seed.

## Deliberate differences from the original authors' setup

- 15 passes (they offered 50) and 128 problems per pass (they allowed up to 512).
- Teacher-forced passes are free here; they counted there.
- Their editor could ask the tool for activations; this one reads them itself with the model's
  own code.
- Their rules were enforced by a sandbox; ours by an edit checker plus instruction, audited
  afterwards.
- A technical check of the evaluator on four practice problems is run before the session and its
  ledger and counter are reset; its outcome is not used to set anything below.

## Endpoints (fixed now)

Finals on MATH-500 (the 500 scored problems; one sample; the seed of this repository's other
finals), run only after the checker has accepted the final weights:

- **Primary — the repository's original protocol** (bare question; `phase3c/rescore.py`), paired
  against stock (7.0%) and the published edit (43.8%).
- **Secondary — the reference protocol** (reference question; last boxed answer; math-verify and
  the repository's matcher on the same boxed answers), paired against the reference arms
  `ref_stock` and `ref_orig`.
- **Verdict, primary protocol:**
  - REPAIR if the final is at least 16.0% and at least stock + 10.0 points (the standing bars).
  - Against the published edit (paired difference with a 95% interval): REACHES IT if within
    ±4.0 points; EXCEEDS if more than +4.0 with the interval excluding 0; FALLS SHORT if less
    than −4.0 with the interval excluding 0; otherwise INCONCLUSIVE.
- Also recorded: the tensors the final edit changed (from its receipt), passes spent, every
  candidate the checker refused and why, and the editor's own diagnosis.

## Predictions (on record before the session)

| # | Prediction | Point estimate |
|---|---|---|
| R1 | The first pass, on the unmodified model, has at least half of its 128 generations cut at the cap. | 0.85 |
| R2 | The final candidate FALLS SHORT of the published edit under the primary protocol. | 0.75 |
| R3 | No candidate of the editor reaches 40% boxed accuracy on the practice pool. | 0.60 |

## What each outcome would change

- R2 wrong (the editor reaches or exceeds the published edit): with the quick fix forbidden and
  in-distribution practice, a small blind editor finds a repair of the hinted frontier editor's
  level. The reported gap was a matter of rules and practice.
- R2 right: a small blind editor, held to the original rules and practising in distribution, does
  not reach it. What remains unseparated: the hint, the editor's size, and the smaller budget
  (15 passes of 128 problems against 50 of up to 512).

## GPU budget

At most 15 practice passes (128 problems, up to 32,768 tokens each) and two finals.
