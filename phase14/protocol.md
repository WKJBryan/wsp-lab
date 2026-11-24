# Pre-registered protocol — Phase 14: the blind editor under the original authors' rules

> Note added 2026-11-24, before the editor session (no pass has been run). The setup, rules,
> endpoints and predictions below are unchanged. Three things are recorded so that the session
> starts from an accurate statement:
> - **The standing bars moved with the stock score.** They are defined against stock, which the
>   final answer checker reads at 8.4% (ISSUES #19), so "at least stock + 10.0 points" means at
>   least 18.4%. The published edit comparator is 48.6%. (The 7.0% and 43.8% written below were
>   the values of the protocol's first re-score.)
> - **The evaluation tool had never been run when this protocol was committed.** It contained a
>   syntax error on the line that prints its result, and looked for the base model's fingerprint
>   in the wrong folder. Both were found by a no-GPU dry run against a fake engine and fixed
>   before the first pass, together with what two independent reviews of the tool found. What
>   the tool does now, none of it changing a rule, an endpoint or a prediction below:
>   - the pass is charged only once the tool holds the GPU lock, so a tool killed while it waits
>     has charged nothing and two tools cannot charge against each other; an interrupted pass is
>     refunded and its ledger row says so;
>   - the texts are written the moment generation returns; every attempt has its own number and
>     its own file (`gen_<attempt>.jsonl`), refunded ones included, and an outcome row in the
>     ledger (MEASURED, FAILED or INTERRUPTED);
>   - a run that fails the batch-position check of `phase3c/runcheck.py` is not a measurement of
>     the candidate (that is the engine's doing, ISSUES #20): the pass is refunded and the summary
>     is kept for the audit but not reported. A run whose generations end in fragments is charged
>     and reported with that finding: fragments are the candidate's own output;
>   - the candidate is fingerprinted again under the lock just before the engine loads it, and the
>     fingerprint covers every file of the directory;
>   - the edit checker treats every non-weight file of the base model as fixed (config, tokenizer,
>     chat template, the shard index), refuses a tensor that appears in two weight files, and
>     issues a receipt for a candidate whose tensors equal the base's although its files differ;
>   - the tool checks its own files against digests recorded when the sandbox was built, and
>     every summary carries the base model's fingerprint and a digest of the tools, to be
>     compared with `phase14/sandbox_manifest.json` after the session;
>   - `--n` must be between 1 and 128.
>   `phase14/test_eval_pass.py` (23 tests) runs the real tool end to end against a fake engine;
>   `phase14/test_save_edit.py` (36 tests) covers the edit checker. One clause was added to
>   `EDITOR_PROMPT.md` so the editor can read the summary: `pass` counts every attempt, refunded
>   ones included; `remaining` is the budget left.
> - **Cost correction.** On this machine one sequence needs about 40 minutes to reach the
>   32,768-token cap, whatever the batch size, so nearly every practice pass takes about 40
>   minutes: fifteen passes are up to about 10 GPU-hours, not the 2 to 3 first estimated, plus
>   two finals of up to 50 minutes each.

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
