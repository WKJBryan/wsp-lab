# Phase 14 results — a blind editor under the original authors' rules does not reach the published edit

**Date:** 2026-11-25/26 (local time) · **Protocol:** `phase14/protocol.md` (pre-registered
2026-11-24, before any editor session and any practice pass; its rules, endpoints and predictions
are unchanged) · **Model:** Qwen3.5-0.8B · Editor that ran: `glm-flash-editor` (GLM-5.3-Flash),
the protocol's fallback, blind; 15 passes available, 2 used.

> **What was registered, and how the run deviated.** The protocol fixed, before the session: one
> blind editor held to the original authors' rules (enforced by the edit checker, `save_edit.py`),
> practising in distribution (128 MATH training problems per pass, the boxed question, thinking
> enabled, at most 32,768 tokens, the same problems and seed every pass), 15 passes, two endpoints
> and three predictions. The run deviated in five ways, disclosed here and in ISSUES #32 and #33.
> **Two attempts:** the protocol's first choice, `mimo-pro-editor`, could not run its session, and
> the fallback, `glm-flash-editor`, ran instead. **Six continuation messages:** during the first
> attempt the host (the agent that runs the editor session) sent six; none named the model, the
> failure or the edits. **A fresh sandbox:** it was built for the second attempt, and the
> four-problem technical check was not repeated. **One rule deviation:** nine of the second
> editor's probe scripts generated a few tokens with the model's own code on fixed sanity strings,
> where the rules allow generation only through the evaluation tool. **The model's name:** both
> editors' notes name the model; the second says it learned the name from the README file shipped
> inside the checkpoint. The task instructions do not name it.

## The verdict

A small blind editor, held to the original authors' rules and practising in distribution, does not
reach the original authors' published edit. Its final edit scores **9.0%** on MATH-500 under the
repository's question, against 8.4% for stock and 48.6% for the published edit. Against stock the
difference is not distinguishable from zero (+0.6 points, 95% interval −2.2 to 3.4); against the
published edit the candidate is 39.6 points behind (interval −44.0 to −35.2). In the protocol's
words: against the published edit, **FALLS SHORT**; REPAIR is not met, since it needs at least
18.4%. Under the reference question the edit reads 10.0% and 10.8% (the repository's matcher and
math-verify) against stock's 7.4% and 7.8%. The edit has the effect it was built for — far more
generations close their thinking (30.8%, against 9.8% for stock) — but a generation that
closes is right far less often, so the net is 3 right answers of 500. All three predictions on
record hold. Two facts bound the reading: the editor that ran was the protocol's fallback model,
and it spent 2 of its 15 passes and stopped by its own choice. It did not converge on the quick fix
the rules forbid (edits to the embedding or output layer and the like); it found a different,
single-tensor edit in layer 0.

## The closing table

MATH-500 under the repository's question (the final answer checker, stamp `#4a1b56a343a4`), one
sample per problem.

| Editor | Hint? | Rules | Passes | MATH-500 | Notes |
|---|---|---|---|---|---|
| Stock (the unedited model) | — | — | — | **8.4%** | baseline; closes its thinking on 9.8%, cut at the cap 90.6% |
| **GLM-5.3-Flash (this phase; the protocol's fallback)** | **no** | **the original authors'** | **2 of 15** | **9.0%** | one tensor, the readout of layer 0's linear attention; **FALLS SHORT** of the published edit by 39.6 points |
| Earlier blind editors (phases 4, 4b, 4c, 10) | no | this repository's | 6 of 10, 7 of 10, 7 of 10, 6 of 50 | 31.6% to 36.2% | all four edits converge on the stop-token row, the quick fix the original authors' rules forbid |
| Original's editor (published edit) | yes | the original authors' | 50 | **48.6%** (our stack) | 2-tensor down_proj surgery |

Under the reference question (last boxed answer; repository matcher / math-verify) the same three
models read 7.4% / 7.8% (stock), 10.0% / 10.8% (this phase's edit) and 46.6% / 51.0% (published
edit).

## The two attempts

### Attempt 1 — `mimo-pro-editor` (MiMo v2.6-pro), the protocol's first choice

Started 2026-11-25 at 21:54. It could not run the session: it ended seven of its turns in about
100 minutes, each time when a stretch of its reasoning closed, and never reached a candidate; its
last three turns produced 0, 2 and 0 tool calls. The host sent six continuation messages (about
22:20, 22:28, 22:57,
23:13, 23:14 and 23:29). The first said: continue autonomously until the finish condition, and do
nothing processor-heavy while a pass is running. The second added a note on how to write the
model's end-of-thinking marker; it did not help and distracted the editor. The third said a turn
ending is harmless and is continued, and to keep only the processor note. The fourth to sixth said
only "continue". None named the model, the failure or the edits.

The editor lost about ten minutes at the start to a shell path problem (its notes: the Git Bash
shell was rewriting the paths it passed to WSL), which it solved itself; no pass was charged for
the failed launches. It spent one pass — the baseline, launched 22:04, measured 22:18 — and
produced no candidate. The host ended the session at 23:32. Its evidence is kept in
`phase14/attempt1_mimo_pro/` (ledger, pass summary, generations, its notes, scripts and outputs).
It also ran processor-heavy teacher-forced reads while its pass was running, which slowed that
pass. Its notes call the model "Qwen3.5-style" without saying where the name came from.
Its baseline pass is identical, generation for generation, to attempt 2's pass 1: the seed is
fixed.

### Attempt 2 — `glm-flash-editor` (GLM-5.3-Flash), the fallback

The protocol's rule for this case: "If that model cannot run, `glm-flash-editor`; whichever ran
is recorded in the results." A fresh sandbox was built at 23:32 with `phase14/make_sandbox.sh`;
its base fingerprint and tool digests equal `phase14/sandbox_manifest.json`. The four-problem
technical check, which the protocol has run before the session (2026-11-25, 01:48), was not
repeated: the tools are byte-identical and the same evaluator had run a full 128-problem pass
minutes earlier. The session started at 23:33 and ended 2026-11-26 at 01:13 — 100 minutes, 490
tool calls, and no host message at all. It was briefed only with the path to
`phase14/EDITOR_PROMPT.md`. It learned the model's name from the README file shipped inside the
checkpoint (its notes say so); the task instructions do not name the model.

## The practice passes

128 problems from the MATH training set (`phase14/practice.json`), under the original authors'
protocol: the boxed question, thinking enabled, at most 32,768 tokens; the same problems and seed
every pass. Both passes pass both validity checks (fragment-tail rate and admission-order split).
The numbers are in `phase14/session/eval_pass_1.json` and `eval_pass_2.json`, the generations in
`gen_1.jsonl.gz` and `gen_2.jsonl.gz`, the ledger in `phase14/session/ledger.md`.

| Pass | Candidate | Correct (boxed) | Closes its thinking | Cut at the cap | Mean tokens | GPU time |
|---|---|---|---|---|---|---|
| 1 | the unedited model | 4.7% (6 of 128) | 5.5% (7) | 94.5% (121) | 31,286 | 775 s |
| 2 | the edit (fingerprint `0a4590253bf4b2f4`) | 7.8% (10) | 26.6% (34) | 79.7% (102) | 27,627 | 682 s |

Paired on the same problems:

| | Pass 2 against pass 1 | Exact McNemar p |
|---|---|---|
| Correct | 7 gained, 3 lost | 0.34 (not distinguishable from chance) |
| Closes its thinking | 29 gained, 2 lost | 4.6e-07 |
| Cut at the cap | 21 no longer cut, 2 newly cut | 6.6e-05 |

Of the 34 generations that close after the edit, 9 are right; of the 7 baseline closers, 6 were.

| Level | Problems | Pass 1 (the unedited model) | Pass 2 (the edit) |
|---|---|---|---|
| 1 | 11 | 0 | 3 |
| 2 | 23 | 4 | 3 |
| 3 | 27 | 2 | 2 |
| 4 | 33 | 0 | 2 |
| 5 | 34 | 0 | 0 |

The editor used 2 of its 15 passes and stopped by its own choice. (Its journal states the
correctness pairs as 4 and 6; counted from the generations they are 3 and 7.)

## The edit

From the receipt (`phase14/session/receipt.json`); re-derived from the weights by the host after
the session, who found exactly this and nothing else.

- One tensor: `model.language_model.layers.0.linear_attn.out_proj.weight` (1024 × 2048), the
  readout of layer 0's linear attention.
- Columns 384 to 511 (its value head 3) multiplied by 0.5.
- Five columns of its head 8 (8 × 128 + 44, 50, 76, 81, 120) set to 0.
- 136,192 values changed; largest change 0.084; norm of the change 2.26.

The edit checker accepted it. No candidate was refused by the checker; the editor dropped several
itself.

## The editor's diagnosis

The editor's own report is `phase14/session/final_report.md`, its journal
`phase14/session/notes.md`. What follows is its reading, from teacher-forced reads and hooks that
cost no pass; the measurements are the editor's.

- **The failure is a verbatim-repetition attractor.** A long generation collapses into a fixed
  "Wait, …" sentence cycle, never emits `</think>` and runs to the 32,768-token cap. At loop
  positions the top-1 next token leads every alternative by 8 to 16 nats, far beyond the
  protocol's presence penalty of 1.5, so decoding cannot escape.
- **The close-then-answer pathway is intact.** `</think>` is reachable (logprob about −2) only
  from specific "ready" contexts ("The final answer is X.", "Okay. Ready.", "This seems
  correct."); in the middle of a derivation it is correctly suppressed (−14 to −25). The 7
  baseline generations that escaped the loop closed, and 6 were right: the loop keeps the model
  from reaching the close, and the close works when reached.
- **Where the loop is produced.** With teacher-forced reads and hooks, zeroing layer 0's
  linear-attention branch (one hook, no weight change) collapses the loop prediction while no
  other single layer matters. Zeroing all six full-attention layers, all 18 linear-attention
  layers or all MLPs breaks it too; in the editor's words, the attractor is "a whole-stack
  resonance seeded by layer 0's readout".
- **Which channels.** In layer 0's `out_proj`, head 3 and five channels of head 8 ({44, 50, 76,
  81, 120}) carry the loop. Closing is carried by two branches, layer 0's linear attention and
  the MLP of layer 23: zeroing either drops the probability of `</think>` from about −2 to −10
  to −13. Within layer 0's readout, the rest of head 8 carries the closing signal, and head 3
  also carries simple arithmetic.

Rejected levers, each measured by the editor:

- Restoring layer 0's decay timescales (24 values of `dt_bias` / `A_log`): the loop margins did
  not move (11.6 → 11.6).
- Zeroing head 3, or quartering it: breaks simple arithmetic ("2 + 2 =" → " 2"); ×0.5 is the
  functional edge.
- Attenuating the six full-attention layers: the model's teacher-forced loss on real successful
  generations rises from 0.577 to 0.821 nats per token at ×0.5 and to 1.032 at ×0.35; at ×0.2 the
  loop goes, but closing and the sanity checks collapse with it.
- Surgery on single heads of layers 1 and 2: negligible (under 0.8 nat).

Its forecast for the scored set: 3 to 8 points above the unedited model, "single-digit to
low-teens".

## The finals on MATH-500

500 problems, one sample each, the repository's seed; run after the edit checker had accepted the
final weights. Both runs pass both validity checks. All numbers are in `phase14/endpoints.json`,
computed by `phase14/endpoints.py` from the collected files (no GPU; it needs the grader's
environment for math-verify);
the runs are `phase3c/p14_final*` and `phase3c/p14_final_ref*`; the reference grader's output is
`phase3c/logs/grade_reference_p14.log`. Both finals carry a re-score by `phase3c/rescore.py`
(`p14_final_rescored.json`, `p14_final_ref_rescored.json`), so they can be re-scored again without
a GPU.

### Primary endpoint — the repository's question

The final answer checker (stamp `#4a1b56a343a4`), paired statistics as in phase 8 (10,000-draw
bootstrap, exact McNemar test).

| Model | Correct | Closes its thinking | Cut at the cap | Mean tokens | GPU time |
|---|---|---|---|---|---|
| Final candidate | **9.0%** (45 of 500) | 30.8% (154) | 75.0% | 25,831 | 37.9 minutes |
| Stock (the unedited model) | 8.4% (42) | 9.8% (49) | 90.6% | 30,108 | 50.3 minutes |
| Published edit | 48.6% (243) | 83.6% | 26.4% | 20,645 | 30.7 minutes |

| Comparison | Difference | 95% interval | Right only under the candidate / only under the other | p | The protocol's word |
|---|---|---|---|---|---|
| Final candidate against stock | +0.6 | −2.2 to 3.4 | 29 / 26 | 0.79 | REPAIR not met: it needs at least 16.0% and at least stock + 10.0 points (18.4%) |
| Final candidate against the published edit | −39.6 | −44.0 to −35.2 | 3 / 201 | 1.1e-55 | **FALLS SHORT** |

Under the two other line rules of ISSUES #31 the same comparison reads: the candidate 8.8% and
9.2%, stock 8.8% and 8.8% (differences 0.0 and +0.4), the published edit 47.6% and 48.4%
(differences −38.8 and −39.2). The words are the same.

Two details of the first row. The upper end of its interval is 3.4 with the problems taken in
sorted order and 3.6 in file order: the bootstrap distribution moves in steps of 0.2 points and
its 97.5th percentile lies just above 3.4. And closing its thinking is not finishing: 42 of the
final candidate's 154 generations that close their thinking still run to the cap afterwards
(stock: 2 of 49).

**What the extra generations that close their thinking contain.** 31 problems close under both models (the
candidate right on 17, the unedited model on 26); 123 close only under the candidate, which is
right on 28; 18 close only under the unedited model, which is right on 16. No generation that
never closes is right under either model. So the edit makes the model finish far more often and
less often correctly; the net is 3 right answers of 500.

### Secondary endpoint — the reference question

Last boxed answer; both graders on the same boxed answers; paired against the reference arms
`ref_stock` and `ref_orig`.

| Model | Repository matcher | math-verify |
|---|---|---|
| Final candidate | **10.0%** (50 of 500) | **10.8%** (54) |
| Stock (`ref_stock`) | 7.4% | 7.8% |
| Published edit (`ref_orig`) | 46.6% | 51.0% |

The final candidate: its last boxed answer is non-blank in 101 of 500 (20.2%; 103 with an extractor
that also reads boxes nested two brace levels deep), and 80 more generations end with an empty
`\boxed{}` copied from the instruction (the reference grader's log counts 181 of 500, 36.2%, with
a box of any kind); cut at the cap 82.4%; mean 27,912 tokens; 41.6 minutes of GPU time.

| Comparison | Repository matcher | math-verify |
|---|---|---|
| Final candidate against stock | +2.6 (95% interval −0.2 to 5.6; 34 problems right only under the candidate, 21 only under stock; p = 0.11) | +3.0 (interval 0.0 to 6.0; 38 against 23; p = 0.072) |
| Final candidate against the published edit | −36.6 (interval −41.0 to −32.2) | −40.2 (interval −44.6 to −35.8) |

The editor's forecast was 3 to 8 points above the unedited model. The measured gains are +0.6
(the repository's question), +2.6 and +3.0 (the reference question): at or below the low end of
its forecast.

## Predictions (on record before the session)

| # | Prediction | Point estimate | Outcome | Result |
|---|---|---|---|---|
| R1 | The first pass, on the unmodified model, has at least half of its 128 generations cut at the cap | 0.85 | 94.5% (121 of 128) were | **holds** |
| R2 | The final candidate FALLS SHORT of the published edit under the primary protocol | 0.75 | −39.6 points, interval −44.0 to −35.2, excluding zero | **holds** |
| R3 | No candidate of the editor reaches 40% boxed accuracy on the practice pool | 0.60 | its only candidate reached 7.8% | **holds** |

## Audit against the manifest and the rules

Done by the host after the session.

- **Tools and model.** The seven tool files carry the digests of `phase14/sandbox_manifest.json`.
  The base model's fingerprint equals the manifest; so does the stock model the sandbox was
  copied from. The practice file equals the repository's copy. The working copy differs from the
  base in that one tensor only. The repository's working tree was not changed.
- **The editor's 52 scripts** (`phase14/session/scripts/`): no internet, no training (no
  gradients, autograd or optimizers), no path outside its own folder.
- **The first editor's 24 scripts and helpers** (`phase14/attempt1_mimo_pro/scripts/` and its work
  folder): no generation outside the evaluation tool, no internet, no training, no path outside
  its own folder.
- **One rule deviation.** Nine probe scripts generate 8 to 24 tokens greedily with the model's own
  code on fixed sanity strings ("The capital of France is", "2 + 2 =", "Q: What is 17 * 24? A:",
  "def fibonacci(n):") to check that a candidate had not broken basic functions — never on
  practice or scored problems. The rule as written allows generation only through the evaluation
  tool (ISSUES #33).

## What it changes

The protocol fixed the reading of each outcome before the session. Its words for this one (R2
right):

> a small blind editor, held to the original rules and practising in distribution, does not reach
> it. What remains unseparated: the hint, the editor's size, and the smaller budget (15 passes of
> 128 problems against 50 of up to 512).

What this run adds to that:

- The editor that ran was the fallback, and it used 2 of its 15 passes by its own choice.
- It did not converge on the quick fix the rules forbid (edits to the embedding or output layer
  and the like). It found a different, single-tensor edit in layer 0 that lets more generations
  close their thinking without making them more often right.
- Under this repository's earlier rules the blind editors' quick fixes reached 31.6% to 36.2%
  (the README's phase table).

## Cost

About 2.0 GPU-hours for the whole item (the protocol estimated 3 to 6): the technical check
161 s, attempt 1's pass 807 s, attempt 2's passes 775 s and 682 s, the two finals 2,274 s and
2,494 s.

## Deviations and limits (also in ISSUES.md #32, #33)

- **The editor of record is the fallback.** The protocol's first choice, `mimo-pro-editor`, could
  not run its session and produced no candidate, so nothing measured here says what that model
  would have scored.
- **Six host messages in the first attempt, and a fresh sandbox for the second** (above,
  ISSUES #32). The second attempt ran with no host message at all.
- **One rule deviation:** nine probe scripts generated outside the evaluation tool, on sanity
  strings only (ISSUES #33).
- **The model's name reached both editors.** The protocol says the editor is not told its name,
  and the task instructions do not name it; the checkpoint ships a README that does, and the
  second editor's notes say it read it.
- **The editor's diagnosis is its own reading**, measured with its own scripts. The audit after the
  session checked the tools, the model, the edit and what the scripts do not do, as listed above.
- **Why the editor stopped after 2 passes is its own account.** Its journal ends: "cand_C is at
  the function edge of the repair family; deeper cuts cost arithmetic/closing."
- One sample per problem, one seed.
- The primary endpoint inherits the checker's line limit (ISSUES #31). The readings under the two
  other line rules are given above and change no word.

## Files added for this phase

- `phase14/results.md` — this file.
- `phase14/endpoints.json` — every number of the two endpoints.
- `phase14/endpoints.py` — computes `endpoints.json` from the collected files; its decision rules
  are tested in `phase14/test_endpoints.py`.
- `phase14/session/` — the ledger, two pass summaries, two generation files, the receipt, the
  editor's final report and journal, 52 scripts, two pass logs.
- `phase14/attempt1_mimo_pro/` — ledger, pass summary, generations, the first editor's notes,
  scripts and outputs.
- `phase3c/p14_final*` and `phase3c/p14_final_ref*` — run, generations, re-score.
- Three logs under `phase3c/logs/`: `p14_final.log`, `p14_final_ref.log`,
  `grade_reference_p14.log`.
