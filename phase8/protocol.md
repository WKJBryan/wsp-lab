# Pre-registered protocol — Phase 8: the missing baselines (is the blind edit a decoding trick?)

Written 2026-11-14 and committed BEFORE any run in this phase and BEFORE anything new is computed
from saved generations. Deviations go to ISSUES.md with reasons. No threshold moves after a run.

## Why this phase

A survey of outside replications (sources at the bottom) raised three questions our record cannot
answer:

1. **The baseline we never ran.** Two independent groups (Toledo; shapor) measured the *unedited*
   model with thinking turned off at 44–50% on MATH-500, above every edited arm we have measured
   (blind single-row edit 27.8%, GLM three-row 25.2%, original edit 39.4%). Our only stock baseline
   is thinking-on (6.0%), the mode in which the model loops.
2. **The control nobody has run.** Every blind editor in phase 4 scaled row 248069 (`</think>`) of
   the tied embedding. Because the embedding is tied to the output layer, that edit raises the
   logit of the stop-thinking token. No published work tests whether a decoding-time boost to that
   token does the same job. (Hu et al. forbade their editor from closing thinking early regardless
   of the problem; NeoCortex found that forcing an answer erased the differences between edits.)
3. **Accuracy on finished answers.** Chalmers and Toledo report accuracy split by finished vs
   cut-off answers. We report termination and accuracy separately, never combined.

## Fixed common settings ("phase-3c final")

Unless an arm says otherwise: vLLM 0.33 under WSL2, `phase3c/problems.json` (500 MATH-500),
1 sample per problem, `SEED = 20261105`, thinking on, temp 0.6 / top-p 0.95 / top-k 20 /
min-p 0 / presence penalty 1.5 / repetition penalty 1.0, cap 32,768, scored by
`phase3c/rescore.py`. These are exactly the settings of every carried run below.

Carried runs (same harness, not re-run; a rerun would only add noise; same policy as phase 4):

| Label | Model | Saved run | MATH-500 |
|---|---|---|---|
| **S** | stock Qwen3.5-0.8B | `phase3c/p3c_stock_rescored.json` | 6.0% |
| **O** | stock + original published edit | `phase3c/p3c_c3_original_rescored.json` | 39.4% |
| **E1** | stock + `embed_tokens` row 248069 × 1.50 (mimo-v2.6-pro, blind) | `phase3c/p4_mimo_pro_rescored.json` | 27.8% |
| **E2** | stock + 3-row rescale (GLM-5.3-Flash, blind) | `phase3c/p4_glmflash_rescored.json` | 25.2% |
| **R** | matched-norm random noise on E2's rows | `phase3c/p4_c2_random_rescored.json` | 2.2% |

E1 is the primary reference edit: one row, one factor, and two editors arrived at it independently.
Its model (`~/wsp3c/p4_mimo_pro` in WSL) is rebuilt from stock with
`phase4/session_mimo_pro/apply_edit.py`, and its sha256 must match the saved copy before any
new arm is built from it. A mismatch voids every arm that uses E1 until resolved.

Known to the author before writing this, disclosed: the summary numbers above, plus each run's
cap-hit rate, `closed` rate and mean tokens. Nothing in Part A below has been computed yet.

## Part A — re-analysis of saved generations (no GPU, run first)

`phase8/analyze_saved.py` over S, O, E1, E2 and R. For each arm:

- accuracy overall; **finished** = `finish == "stop"`; accuracy among finished; accuracy among
  cut-off (answers scraped from a `\boxed{}` written inside unfinished thinking); mean tokens of
  finished generations.
- paired exact McNemar test and paired-bootstrap 95% CI (10,000 resamples over problems,
  numpy seed 0) for: E1 vs S, E1 vs O, E2 vs E1, O vs S.
- **Second grader:** math-verify in the WSL venv, run on every row's gold answer and full output.
  Any exception is counted and the row is marked ungradable. There is no silent fallback to string
  comparison (that is the failure in NeoCortex's erratum, and math-verify already broke for us on
  Windows, ISSUES #8–9). Report agreement with `rescore.py` per arm. If agreement is below 97% on
  any arm, every verdict in this phase is reported under both graders.

| # | prediction (fixed now) |
|---|---|
| **A1** | Accuracy among finished answers: E1 ≤ O − 10 points (the blind edit stops thinking earlier and its finished answers are worse) |
| **A2** | Mean tokens of finished generations: E1 < O |
| **A3** | At least 2 points of E1's 27.8% comes from cut-off generations (a `\boxed{}` written inside unfinished thinking) |

Reading: if A1 holds, the write-up must say that the blind edit buys termination at the cost of
answer quality, whatever Parts B–D show.

## Part B — Qwen's own baselines, thinking off (GPU, short runs)

Settings copied verbatim from the Qwen3.5-0.8B model card (read 2026-11-14), non-thinking text
mode: temperature 1.0, top_p 1.00, top_k 20, min_p 0.0, presence_penalty 2.0,
repetition_penalty 1.0, max output 32,768. Thinking disabled with
`chat_template_kwargs={"enable_thinking": False}`. The rendered prompt for problem 0 is saved
for each mode, to record exactly what the template inserts.

**Scoring fix, fixed now:** `rescore.py` returns no answer when there is no `\boxed{}` and no
`</think>`. Thinking-off outputs may contain no `</think>`, so a `--no-think` flag treats the whole
output as the answer section. The `\boxed{}` path is unchanged. The flag applies to B-arms only.

| Arm | Model | Decoding |
|---|---|---|
| **B1** | S | card non-thinking settings |
| **B2** | S | thinking off, phase-3c decoding (isolates the mode switch from the sampling change) |
| **B3** | E1 | card non-thinking settings (does the edit hurt the mode that already works?) |

| # | prediction (fixed now) |
|---|---|
| **B-1** | B1 lands in 40–52% (Toledo 44.0–49.7, shapor 45.2) |
| **B-2** | B3 is within ±4 points of B1 |

Reading, fixed now whatever the numbers: if B1 > E1, the phase-4 headline is restated as "the blind
edit repairs thinking mode; it does not beat turning thinking off". If B1 > O as well, the same
sentence is added for the original edit, as Toledo and shapor already report.

## Part C — decoding-only controls on stock (thinking on, phase-3c final)

**C1 — constant logit bias on `</think>`.** `SamplingParams(logit_bias={248069: b})`, weights
untouched.
- Integrity probe first, on the 4 phase-3c probe problems: b = 0 must reproduce stock text byte for
  byte. b = +30 must close thinking within the first 5 generated tokens on all 4. If vLLM 0.33
  ignores `logit_bias`, the fallback is a logits processor that adds b to token 248069. The fallback
  is logged as a deviation and probed the same way.
- Tuning budget: **7 practice passes**, the same as each mimo arm had. Harness `phase8/eval_pass8.py`
  is `phase4/eval_pass.py` plus a `--logit-bias` flag, with the same practice set
  (`phase4/practice.json`), cap 8192, counter file and pre-logged hypothesis. No offline
  generation is allowed: every evaluation channel is budgeted (the lesson of ISSUES #12).
- Fixed opening grid: b = +2, +4, +6. The remaining 4 passes are free choices, each pre-logged.
- Selection rule, fixed now: highest practice accuracy, ties going to the smaller b. The author sees
  practice accuracy (not blind). That helps the control, which is the conservative direction for any
  claim that the weight edit is special.
- Final: one MATH-500 run at the chosen b.

**C2 — budget forcing (s1-style), from saved generations.** Applied to S, O and E1. For each saved
generation, re-tokenize the text with the model tokenizer:
- `</think>` within the first 30,720 tokens → kept unchanged.
- otherwise → truncate to 30,720 tokens, append the cue below, and generate at most 2,048 more
  tokens with phase-3c decoding and seed. The prompt is the thinking-on chat prompt, then the
  truncated text, then the cue.
- Cue, fixed now verbatim: `"\n\nConsidering the limited time by the user, I have to give the
  solution based on the thinking directly now.\n</think>\n\n"`. (It is Qwen3's documented
  budget-exceeded cue. The Qwen3.5 card documents none.)
- Disclosed risk: decoding then re-encoding may not reproduce the original token ids. Report the
  share of saved generations whose re-encoded length differs from the saved `n_tokens`.

| # | prediction (fixed now) |
|---|---|
| **C-1** | C1 MATCHES E1 (rule below) |
| **C-2** | Forced S ≥ 16.0% |
| **C-3** | Forcing shrinks the O − E1 gap: (forced O − forced E1) < (39.4 − 27.8 = 11.6) |

## Part D — split the single-row edit into its two halves

The tied row acts in two places: on the **output** side (the `</think>` logit is h·e, so it becomes
exactly 1.5× when there is no output bias; checked before building) and on the **input** side (the
embedding read back once `</think>` has been written, which shapes the answer section). An untied
copy separates the two. `phase8/build_untied.py` takes stock, sets `tie_word_embeddings: false` at
both the top level and in `text_config`, and adds `lm_head.weight` as a copy of
`model.language_model.embed_tokens.weight` (exact key name confirmed against vLLM's loader before
building).

- **U0:** untied, nothing scaled. On the 4 probe problems it must reproduce stock text byte for
  byte, or Part D is void (logged) and only the output half is tested, through a logits processor
  that multiplies logit 248069 by 1.5.
- **D-out:** `lm_head` row 248069 × 1.5, embedding unchanged. This equals a decoding-time ×1.5 on
  the stop-thinking logit.
- **D-in:** embedding row 248069 × 1.5, `lm_head` unchanged.
- The MTP head (`mtp_num_hidden_layers: 1`) is unused without speculative decoding. Confirmed off in
  the run config.

Both arms get a MATH-500 final.

| # | prediction (fixed now) |
|---|---|
| **D-1** | D-out MATCHES E1 |
| **D-2** | D-in MATCHES S |

## Decision rules (every comparison paired over the 500 problems)

For arm X against reference Y: Δ = acc(X) − acc(Y), 95% CI by paired bootstrap (as in Part A).
The 4.0-point margin is about 2 standard errors of a single arm at 28%, n = 500.

- **MATCHES:** |Δ| ≤ 4.0
- **FALLS SHORT:** Δ < −4.0 and the CI's upper end < 0
- **EXCEEDS:** Δ > 4.0 and the CI's lower end > 0
- otherwise **INCONCLUSIVE**

Verdicts must hold under both graders when the Part-A agreement rule triggers.

Combined readings, fixed now:

| C1 vs E1 | D-out vs E1 | What the write-up says |
|---|---|---|
| MATCHES | MATCHES | The blind edit is a decoding-time boost to the stop-thinking token, stored in the weights. The "three editors converged" result is restated as convergence on a decoding fix. |
| FALLS SHORT | MATCHES | The edit is decoding-level but state-dependent: a multiplicative boost does what a constant bias cannot. |
| any | FALLS SHORT | The input side carries part of the effect, so the edit is more than a decoding tweak. D-in then says how much. |
| any | INCONCLUSIVE | Reported as such. No mechanism claim. |

## Order, budget, reporting

1. Part A (free), then the integrity probes (U0, b = 0 / b = +30, E1 sha256), then B, C, D.
2. GPU estimate on the RTX 5090: B arms are short (thinking off); C1 is 7 practice passes plus one
   final; C2 regenerates only the forced rows of 3 runs; D is 2 finals. About 3–4 GPU-hours.
   No editor models and no API spend in this phase.
3. Results go in `phase8/results.md`: every prediction marked HIT or MISS, every number from a fresh
   run with its JSON committed, generations gzipped as in phase 3c.

## Out of scope (named so nobody mistakes them for forgotten)

- Negative control on a model that does not loop (Gemma 3 270M, with the blind editors). Candidate
  for phase 9.
- IFEval and other cross-task transfer of E1 and E2.
- Qwen's 81,920-token thinking baseline. Toledo reports 22.3%; cited, not re-run.
- Multiple seeds and n = 4 finals. n = 1 keeps every arm paired with the carried runs; SE is about
  2 points (disclosed).
- Stacking E1 on top of O (Grover's stacking result suggests it is worth a test).

## Files to write after this protocol is committed

`phase8/analyze_saved.py` (Part A) · `phase8/run_vllm8.py` (`phase3c/run_vllm.py` plus `--no-think`,
`--card-nonthink`, `--logit-bias`, `--force-from <run>`) · `phase8/eval_pass8.py` ·
`phase8/build_untied.py` · `--no-think` flag in `phase3c/rescore.py`.

## Sources this phase responds to

- Toledo, re-graded baselines: https://carlostoledo.co/reports/weight-space-programming.html
- shapor, thinking-off control: https://github.com/shapor/weight-space-experiments
- Hu et al., Exploring Weight-Space Programming (arXiv 2611.09382): https://arxiv.org/html/2611.09382
- NeoCortex0815, answer forcing and grader erratum: https://github.com/NeoCortex0815/knowing-is-not-committing
- Chalmers, finished vs cut-off split: https://github.com/kyle-chalmers/weight-space-programming
- Qwen3.5-0.8B model card (sampling settings, tied embeddings): https://huggingface.co/Qwen/Qwen3.5-0.8B
