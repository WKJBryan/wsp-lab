# Phase 15 results — a stronger blind editor under the original authors' rules reaches the published edit (within the protocol's band: 265 of 500 against 271)

**Date:** 2026-11-26/27 (local time) · **Protocol:** `phase15/protocol.md` (pre-registered
2026-11-26, before the editor session and before any practice pass of this phase; its notes were
written before the session and changed no rule, endpoint or prediction) · **Model:** Qwen3.5-0.8B
(`Qwen3_5ForConditionalGeneration`) · Editor that ran: GLM-5.4, blind, through the harness's
built-in general helper (the registered `glm54-editor` type could not be loaded into the running
chat); 15 passes available, 4 used.

> **What was registered, and how the run deviated.** The protocol fixed, before the session: one
> blind editor (GLM-5.4) held to the original authors' rules (enforced by the edit checker,
> `save_edit.py`), practising in distribution (128 MATH training problems per pass, the boxed
> question, thinking enabled, at most 32,768 tokens, the same problems and seed every pass), 15
> passes, the same briefing file the phase-14 editor received (`phase14/EDITOR_PROMPT.md`, byte for
> byte), two endpoints, one exploratory comparison (against the phase-14 editor's final) and five
> predictions; everything else is inherited from `phase14/protocol.md`. Written before the session,
> it also records a four-problem technical check in the fresh sandbox (2026-11-26, 22:51; its
> outcome sets nothing) and that the sandbox's tools carry the answer checker's third rule, so every
> comparator is read under that checker. The run deviated as follows, disclosed here, in ISSUES #35 to #38 and under
> audit below. **The editor's helper:** the registered `glm54-editor` type cannot be loaded into a
> chat that is already running, so the editor ran through the harness's built-in general helper,
> which has more tools than the editor types; its tools were restricted by instruction and audited
> afterwards. **One host message:** the host (the agent that runs the editor session) sent one, the
> word "continue", after a connection failure on the host stopped the session; nothing else was
> sent. **The first part of the session has no harness record:** the resume overwrote it. **The
> model's family reaches the editor through the checkpoint's own files**; the task instructions do
> not name it — the same situation as phase 14 (ISSUES #33). **Two smaller ones:** a byte-code
> cache folder appeared under the sandbox's tools folder, and the editor saw a folder of the
> technical check's evidence in its first directory listing (it states that it did not open it).

## The verdict

A stronger blind editor (GLM-5.4), held to the original authors' rules and practising in
distribution (128 MATH training problems per pass), reaches the original authors' published edit.
Its final edit scores **265 of 500 (53.0%)** on MATH-500 under the repository's question, against
the published edit's 271 (54.2%) and stock's 47 (9.4%). In the protocol's words: against the
published edit, **REACHES IT** — the difference is −1.2 points, inside the protocol's band of
±4.0, with a 95% interval of −5.0 to 2.6 — and **REPAIR is met**: against stock the difference is
+43.6 points (95% interval 39.2 to 48.0; 218 problems right only under the candidate, none only
under stock), where the rule needs at least 80 of 500 right and at least stock + 10.0 points (50 of
500). Under the reference question (the original researchers' kind of question; the last boxed
answer) the edit comes out above the published edit: 280 of 500 (56.0%) against 258 (51.6%) under
the repository's matcher (+4.4 points, 95% interval 0.8 to 8.0) and 282 of 500 (56.4%) against 259
(51.8%) under math-verify (+4.6 points, interval 1.0 to 8.2). Against the phase-14 editor's final
(55 of 500, 11.0%) it is +42.0 points (interval 37.4 to 46.6). The edit has the effect it was built
for: 496 of 500 (99.2%) of its generations close their thinking, against 49 (9.8%) for stock, and
all 265 right answers come from generations that closed.

So the outcome is the first branch of the protocol's "What each outcome would change": the gap left
by phase 14 was a matter of the editor's strength, not of the hint or of the budget. The author's
own predictions said otherwise: P2 (FALLS SHORT) and P3 (REPAIR not met) were wrong; P1, P4 and P5
hold.

Five things bound the reading:

- **One session and one seed** for the editor; one sample per problem for every scored run.
- **"Reaches it" means within 4 points.** The measured difference is −1.2 points and its 95%
  interval runs from −5.0 to 2.6: the two edits cannot be told apart under the repository's
  question.
- **The mechanism is different from the published edit's.** The published edit removes a measured
  loop direction in layers 15 and 19 so that the loop does not form. This edit is a conditional
  loop-breaker: it lets the loop form and ends the thinking where the model is detected repeating
  itself. It is within the original authors' rule as written ("… close it early regardless of the
  problem") and was accepted by the edit checker; whether the original authors count it as a repair
  of the same kind is theirs to say.
- **The smaller budget.** The editor had the budget of phase 14, 15 passes of 128 problems and not
  50 of up to 512, and used 4 of them.
- **About a sixth of the problems get no answer.** 87 of 500 (17.4%) are cut at the cap, 83 of them
  after the thinking closed: the model falls back into the loop while writing its answer.

And the run was not clean: the editor ran through a helper with more tools than the editor types,
restricted by instruction; its session was stopped once by a connection failure and followed by one
host message; and the harness's record of the first part of the session is lost (below). (Every
number in this file is the final answer checker's reading, stamp `#d74fdebd340f`, ISSUES #19, #30,
#31; so is every comparator's.)

## The closing table

MATH-500 under the repository's question (the final answer checker, stamp `#d74fdebd340f`), one
sample per problem.

| Editor | Hint? | Rules | Passes | MATH-500 | Notes |
|---|---|---|---|---|---|
| Stock (the unedited model) | — | — | — | 47 of 500 (9.4%) | baseline; closes its thinking on 49 of 500 (9.8%), cut at the cap 453 of 500 (90.6%) |
| **GLM-5.4 (this phase)** | **no** | **the original authors'** | **4 of 15** | **265 of 500 (53.0%)** | a conditional loop-breaker, three tensors in layer 14's MLP; **REACHES IT**: 1.2 points below the published edit (95% interval −5.0 to 2.6); REPAIR met |
| GLM-5.3-Flash (phase 14, the protocol's fallback there) | no | the original authors' | 2 of 15 | 55 of 500 (11.0%) | one tensor, the readout of layer 0's linear attention; 42.0 points below this phase's edit (interval 37.4 to 46.6); its record is `phase14/results.md` |
| Original's editor (published edit) | yes | the original authors' | 50 | 271 of 500 (54.2%) | removes a measured loop direction in layers 15 and 19 so that the loop does not form |

Under the reference question (last boxed answer; repository matcher / math-verify) stock reads 44 of
500 (8.8%) / 44 of 500 (8.8%), this phase's edit 280 of 500 (56.0%) / 282 of 500 (56.4%) and the
published edit 258 of 500 (51.6%) / 259 of 500 (51.8%).

## The session

### The editor, and how it ran

The protocol registered a `glm54-editor` helper type. The harness fixes its helper types when a
chat starts, and that type was defined afterwards, so it cannot be loaded into the chat that runs
the session; the first attempt to start it was refused and nothing ran (the protocol's note of
2026-11-26, 22:58, written before the session). The editor therefore ran through the harness's
built-in general helper, on GLM-5.4. A probe before the session (no tools used) showed what that
helper is handed at its start: the user's general working rules, which say nothing about this
project, the fault or mathematics, and a list of tool names; no project notes. Its briefing was the
editor types' four operating rules, word for word, the sentence that its tools are exactly those of
the editor types (Read, Glob, Grep, Write, Edit, Bash; no web tools, no other helpers), and the path
of `phase14/EDITOR_PROMPT.md` — the same briefing file the phase-14 editor received, byte for byte.
The helper has more tools than that, so the restriction was enforced by instruction and audited
afterwards (below).

The sandbox was rebuilt with `phase14/make_sandbox.sh`, the phase-14 sandbox having been moved
aside whole (renamed, not deleted). The four-problem technical check ran in it before the session
(2026-11-26, 22:51): exit status 0, all four generations ran to the cap, and the summary's base
fingerprint and tools digest equal `phase15/sandbox_manifest.json`; its ledger and counter were
then reset, so the sandbox held no pass when the session began.

### The connection failure and the one host message

- The session started 2026-11-26 at 22:59 (local time). Pass 1, on the unedited model, ran 23:00:54
  to 23:14:21.
- **The connection failure.** Between 00:32 and 00:44 on 2026-11-27 the session stopped with the
  error "Cannot connect to API: getaddrinfo EAI_AGAIN" for the model service: a failure on the
  host, not the editor's choice (its last recorded step began at 00:31:53). Noticed at 00:45: 1 of
  15 passes charged, no receipt, no final report, no process of the editor running, tools and
  practice file unchanged.
- **The one host message.** At 00:46 the host sent the word "continue", and nothing else. (The
  protocol: the host sends no message unless a turn ends before the finish condition, and then a
  neutral "continue" and nothing else.) The editor resumed with its context. No other host message
  was sent; `phase15/session/host_messages.md` is the record.
- Pass 2 ran 01:02:25 to 01:09:20, pass 3 01:54:38 to 02:01:13 and pass 4 02:26:00 to 02:31:37.
- The editor ended the session by itself at 02:39 with its final report; its last file was written
  at 02:39. It ran the edit checker on its final state once more at 02:37:45 (its receipt is noted
  "final state confirmation").
- 4 of 15 passes charged, none refunded, no candidate refused.

### Length and use

About 3 h 40 min from the start to the final report, of which about 12 minutes were lost to the
interruption. The harness's record of the resumed part (00:46 to 02:39) gives 114 minutes, 169 tool
calls and about 77 million tokens (229 thousand of them output). The record of the first part was
overwritten by the resume, so its tool calls and tokens are not known. For scale, the phase-14
editor's whole session took 100 minutes, 490 tool calls and about 86 million tokens.

## The practice passes

128 problems from the MATH training set (`phase14/practice.json`), under the original authors'
protocol: the boxed question, thinking enabled, at most 32,768 tokens; the same problems and seed
every pass. All four passes pass both validity checks (fragment-tail rate and admission-order
split). The numbers are the tool's summaries, `phase15/session/eval_pass_1.json` to
`eval_pass_4.json`; the generations are in `gen_1.jsonl.gz` to `gen_4.jsonl.gz`, the ledger in
`phase15/session/ledger.md`. The sandbox's tools carry the answer checker's third rule (the
protocol's note; `phase15/sandbox_manifest.json`), so these summaries are already the final answer
checker's reading.

| Pass | State | Right, boxed rule | Right, repository's reading | Closes its thinking | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|---|
| 1 | the unedited model | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 121 of 128 (94.5%) | 31,286 | 806.8 s |
| 2 | the first switch (repetition) | 59 (46.1%) | 61 (47.7%) | 120 (93.8%) | 44 (34.4%) | 18,624 | 415.9 s |
| 3 | both switches — **the final edit** | 68 (53.1%) | 70 (54.7%) | 126 (98.4%) | 32 (25.0%) | 17,769 | 394.3 s |
| 4 | both, a lower threshold on the first switch | 69 (53.9%) | 70 (54.7%) | 127 (99.2%) | 16 (12.5%) | 15,560 | 336.9 s |

Paired on the same problems, boxed rule: pass 2 against pass 1, 53 gained and 1 lost; pass 3
against pass 2, 15 gained and 6 lost; pass 4 against pass 3, 10 gained and 9 lost; pass 3 against
pass 1, 61 gained and none lost.

| Level | Problems | Pass 1 (the unedited model) | Pass 2 | Pass 3 (the final edit) | Pass 4 |
|---|---|---|---|---|---|
| 1 | 11 | 0 | 9 | 10 | 9 |
| 2 | 23 | 4 | 14 | 17 | 19 |
| 3 | 27 | 3 | 15 | 16 | 18 |
| 4 | 33 | 0 | 16 | 17 | 17 |
| 5 | 34 | 0 | 5 | 8 | 6 |

(Right by level, boxed rule.)

Fingerprints: pass-2 state `071c042292c351ee`, pass-3 state (the final) `a3de53261dbcd9be`,
pass-4 state `cde4c2c5907c1a00`. The editor kept the pass-3 state, by a rule it fixed before pass 4
(the challenger had to be clearly better; one problem more is noise); its ledger's entry on pass 4:
"NOT kept: not clearly better than g2 (+1 problem, within noise)". Its own caveat: the practice
scores of the edited states are single draws, and pass-to-pass differences of several points are
mostly what it calls trajectory reshuffle.

Its forecast for the scored set, written before any scored run: 45% to 60% right, central estimate
near 50%, against about 3% to 9% for the unedited model. Measured: 53.0% (the repository's
question) and 56.0% / 56.4% (the reference question).

The editor used 4 of its 15 passes, none refunded; the checker refused no candidate.

## The edit

From its receipt (`phase15/session/receipts/`, the one noted "final state confirmation");
re-derived from the weights by the host after the session, who found exactly this.

- All in layer 14's MLP, in two neurons (1439 and 2910): three tensors, 6,143 values changed in
  all, out of 873 million; the other 485 tensors and every non-weight file are identical to the
  base model.

| Tensor | What changed | Values changed | Largest change | Size of the change (norm) |
|---|---|---|---|---|
| `model.language_model.layers.14.mlp.gate_proj.weight` | rows 1439 and 2910 | 2,047 | 0.530 | 4.74 |
| `model.language_model.layers.14.mlp.up_proj.weight` | rows 1439 and 2910 | 2,048 | 0.027 | 0.345 |
| `model.language_model.layers.14.mlp.down_proj.weight` | columns 1439 and 2910 | 2,048 | 0.795 | 5.86 |

**How it works** (the editor's description). Each of the two neurons now computes
`silu((d·x − θ·w·x) / s) · (w·x)` and adds that, times `c·v`, to the layer's output. `v` is the
unit output direction of the model's end-of-thinking token at layer 14; `w` is a constant
direction; `d` is a direction that separates line starts inside word-for-word repetition from line
starts in fresh text (first neuron: θ = 425, s = 20, c = 5.5); the second neuron uses a direction
for a runaway numbered list (θ = 5, s = 10, c = 2). So each neuron is a switch that is silent
everywhere except where the model is detected repeating itself, and where it fires it makes the
model end its thinking. The directions are means, covariances and one regularised linear
discriminant, computed in one step from the model's own generations on the 128 practice problems:
no backward pass, no optimizer, no iterative fitting to a loss.

**The rule, and the difference from the published edit.** The original authors' instruction put
"making the model skip its thinking or close it early regardless of the problem" out of scope.
This edit ends the thinking only where repetition or a runaway list is detected, so it is inside
that rule as written, and the edit checker accepted it. It is not the published edit's mechanism:
that one removes a measured loop direction from two weight tables (layers 15 and 19) so that the
loop does not form, and the model then thinks properly and stops by itself. This one lets the loop
form and breaks it. Call it "a conditional loop-breaker". Whether the original authors count it as
a repair of the same kind is theirs to say.

The edit checker accepted all three edited states the editor submitted (three receipts); no
candidate was refused.

## The editor's diagnosis

The editor's own report is `phase15/session/work/final_report.md`, its journal
`phase15/session/work/notes.md` (it titled both "Phase 14": it was briefed with the phase-14
instructions). What follows is its reading, from teacher-forced reads that cost no pass; the
measurements are the editor's.

- **Nothing is broken in the weights; the repetition is the model's own behaviour.** Per-tensor
  weight statistics are smooth across layers, and removing any attention head at runtime (no
  weight change) raises the clean-text loss for every head but one unused one. Its decisive
  check: the weight file's SHA-256 equals the hash recorded at download time, which it found in
  the checkpoint's own files. (The host's audit confirms that the reference it compared with is the
  download record inside the checkpoint.)
- **The failure is an exact repeated cycle.** 113 of 128 baseline generations end in an exact
  repeated cycle (period 6 to 322 tokens) that begins after a median of about 7,000 generated
  tokens and never ends; the cycles are usually "Wait, …" lines. Inside a loop the cycle's next
  token leads by about 15 raw logits; the protocol's presence penalty is a flat 1.5, so it cannot
  break a loop.
- **A single direction carries the signal.** In layer 14's MLP input, one direction (the mean at
  line starts inside loops minus the mean at line starts in fresh text) separates the two with AUC
  about 0.98 on held-out problems: in-loop line starts score 454 to 480, the highest legitimate
  content 311.
- **How much the switch must write.** A unit write along the output direction of `</think>` at
  layer 14 raises that token's logit by about +7; in loops `</think>` trails the top token by 25 to
  30 logits, so a write of 4 to 6 units flips it. After a forced `</think>` the detector score
  collapses, so the switch goes silent and the model starts a normal answer.
- **Why two switches.** In pass 2 (the first switch alone) 84 generations ended (59 right) and 44
  were cut at the cap, none right: 36 closed their thinking and then fell back into the same loop
  while writing the answer; 8 never closed, because they were runaway numbered lists (each line
  differs by an increment), not word-for-word repeats. The second switch is for those.
- **Not repaired: the loop during the answer.** After the thinking is ended, the model falls back
  into the same loop while writing its answer. The editor tested single-token levers at the stuck
  positions: none is naturally followed by `\boxed{` (the weak link is always 0.2 or less), the
  model's guess at those positions is mostly wrong (about 2 of 30), and the detector collapses one
  position after any forced token, so chains cannot be driven. In its reading these are problems
  the model cannot solve, so it left them alone: on the practice set that left about a third of
  the problems without an answer. On the scored set 87 of 500 (17.4%) are cut at the cap, 83 of
  them after the thinking closed (the host's count).
- **Its own account of the second switch's size.** Tracing which switch closed each pass-3
  generation, it found the second switch itself closed only 1 to 2; most of the change from pass 2
  to pass 3 is trajectory reshuffle. It kept the switch on mechanism evidence.
- **Its clean-text cost.** The prediction loss on ordinary reasoning text is unchanged to four
  decimals (0.3222 before and after, its measure). It states that it never ran MATH-500 or read
  its problems: all its directions come from the 128 practice problems.

## The finals on MATH-500

500 problems, one sample each, the repository's seed; run after the edit checker had accepted the
final weights. The repository's question ran 02:42:10 to 03:10:28 on 2026-11-27 and the reference
question 03:10:28 to 03:42:37; the reference grader took 4 seconds. Both scored runs pass both
validity checks (fragment tails 0.0%; no dependence on the request's position in the batch). All
numbers are in `phase15/endpoints.json`, computed by `phase15/compute_endpoints.py`, which applies
`phase14/endpoints.py` to this phase's runs and adds the exploratory comparison (no GPU; it needs
the grader's environment for math-verify); the runs are `phase3c/p15_final*` and
`phase3c/p15_final_ref*`, each with its re-score by `phase3c/rescore.py`; the reference grader's
output is `phase3c/logs/grade_reference_p15.log`. Every re-scored file carries the stamp
`phase3c/rescore.py+phase3b/scorer.py#d74fdebd340f (ISSUES #19, #30, #31)`.

### Primary endpoint — the repository's question

The final answer checker (stamp `#d74fdebd340f`), paired statistics as in phase 8 (10,000-draw
bootstrap, seed 0, problems in sorted order; exact McNemar test). All three runs pass both
validity checks.

| Model | Right | Closes its thinking | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|
| Final candidate | **265 of 500 (53.0%)** | 496 of 500 (99.2%) | 87 of 500 (17.4%) | 16,308 | 1,609.1 s |
| Stock (the unedited model) | 47 of 500 (9.4%) | 49 of 500 (9.8%) | 453 of 500 (90.6%) | 30,108 | 3,016.2 s |
| Published edit | 271 of 500 (54.2%) | 418 of 500 (83.6%) | 132 of 500 (26.4%) | 20,645 | 1,839.0 s |

| Comparison | Difference | 95% interval | Right only under the candidate / only under the other | p | The protocol's word |
|---|---|---|---|---|---|
| Final candidate against stock | +43.6 points | 39.2 to 48.0 | 218 / 0 | 4.75e-66 | REPAIR met: it needs at least 80 of 500 right and at least stock + 10.0 points (50 of 500); the candidate has 265 and +218 |
| Final candidate against the published edit | −1.2 points | −5.0 to 2.6 | 42 / 48 | 0.60 | **REACHES IT** (the band is ±4.0 points) |

- 223 problems are right under both the candidate and the published edit; 42 only under the
  candidate; 48 only under the published edit.
- All 265 right answers of the candidate come from generations that closed their thinking: 204
  from a line that carries "answer is" and the like, 61 from a concluding last line.
- Closing its thinking is not finishing. Of the candidate's 496 generations that close, 83 are
  still cut at the cap afterwards (the model loops again while writing its answer); 4 never close,
  and none of those is right. Stock: 2 of its 49 closers are cut afterwards; the published edit:
  51 of 418.

### Secondary endpoint — the reference question

The original researchers' kind of question; last boxed answer; both graders on the same boxed
answers; paired against the reference arms `ref_stock` and `ref_orig`.

| Model | Repository matcher | math-verify | Boxed answers | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|
| Final candidate (`p15_final_ref`) | **280 of 500 (56.0%)** | **282 of 500 (56.4%)** | 325 of 500 | 25.2% | 17,875 | 1,866.0 s |
| Stock (`ref_stock`) | 44 of 500 (8.8%) | 44 of 500 (8.8%) | 44 of 500 | 93.2% | 30,991 | 2,871.9 s |
| Published edit (`ref_orig`) | 258 of 500 (51.6%) | 259 of 500 (51.8%) | 320 of 500 | 29.0% | 21,796 | 1,788.0 s |

| Comparison | Repository matcher | math-verify |
|---|---|---|
| Final candidate against stock | +47.2 points (95% interval 42.8 to 51.6; 236 problems right only under the candidate, 0 only under stock; p = 1.81e-71) | +47.6 points (interval 43.2 to 52.0; 238 against 0; p = 4.53e-72) |
| Final candidate against the published edit | **+4.4 points (interval 0.8 to 8.0; 54 against 32; p = 0.023)** | **+4.6 points (interval 1.0 to 8.2; 55 against 32; p = 0.018)** |

No protocol word attaches to the reference comparisons; under both graders the interval excludes
zero. Read by the repository's usual rule (which credits answers stated outside the box), the same
reference run gives 291 of 500 (58.2%).

The editor's forecast was 45% to 60% right, central estimate near 50%. Measured: 53.0% (the
repository's question) and 56.0% / 56.4% (the reference question): inside its band.

### Exploratory — against the phase-14 editor's final

New in this phase's protocol; no protocol word. The phase-14 editor's final (`p14_final`, the
GLM-5.3-Flash editor's edit) reads 55 of 500 (11.0%) under the repository's question; it closes
its thinking on 154 of 500 (30.8%), and 375 of 500 are cut at the cap.

| Comparison | Difference | 95% interval | Right only under this phase's candidate / only under the phase-14 final | p |
|---|---|---|---|---|
| Final candidate against the phase-14 editor's final | +42.0 points | 37.4 to 46.6 | 215 / 5 | 4.98e-57 |

The protocol's own limit on this comparison: it is exploratory, one session each and one seed
each, so it cannot separate the editors' strength from the luck of a session.

Four other exploratory comparisons of the final candidate (the repository's question; no protocol
word; in `phase15/endpoints.json` under `other_exploratory_runs` and `other_exploratory_pairs`, each
of the four runs held to the same checks as the endpoints):

| Against | Difference | 95% interval | Right only under the candidate / only under the other | p |
|---|---|---|---|---|
| The best blind quick fix (MiMo-flash, phase 4b; `p4_mimo`; 211 of 500, 42.2%) | +10.8 points | 6.8 to 14.8 | 82 / 28 | 2.52e-07 |
| The 50-pass edit (phase 10; `p10_fullbudget`; 208 of 500, 41.6%) | +11.4 points | 7.4 to 15.4 | 82 / 25 | 2.91e-08 |
| The published edit with our repair at full strength (phase 7; `p7_stack_full_v2`; 261 of 500, 52.2%) | +0.8 points | −3.0 to 4.6 | 47 / 43 | 0.752 |
| Our map repair (phase 7; `p7_uperp_on_loop`; 182 of 500, 36.4%) | +16.6 points | 12.4 to 20.8 | 104 / 21 | 2.1e-14 |

So, with no protocol word: the candidate is above the best blind quick fix and above the 50-pass
edit (both intervals exclude zero), and not distinguishable from the published edit with our repair
at full strength.

## Predictions (on record before the session)

| # | Prediction | Point estimate | Outcome | Result |
|---|---|---|---|---|
| P1 | The first pass, on the unmodified model, has at least half of its 128 generations cut at the cap | 0.95 | 121 of 128 (94.5%) were | holds |
| P2 | The final candidate FALLS SHORT of the published edit under the primary protocol | 0.80 | −1.2 points, interval −5.0 to 2.6: REACHES IT | **wrong** |
| P3 | REPAIR is not met | 0.65 | met: 265 of 500 right, 218 over stock | **wrong** |
| P4 | The editor ships an edit: the final weights differ from the base model in at least one tensor | 0.90 | three edited states, the last accepted with its receipt | holds |
| P5 | The final candidate scores above the phase-14 editor's final by a paired difference whose 95% interval excludes zero | 0.40 | +42.0 points, interval 37.4 to 46.6 | holds |

## Audit against the manifest and the rules

Done by the host after the session.

- **Tools, model, edit.** The seven tool files carry the digests of `phase15/sandbox_manifest.json`;
  the base model's fingerprint equals the manifest's; the final weights (fingerprint
  `a3de53261dbcd9be`) carry their receipt; three receipts in all, no refusal. The edit was
  re-derived from the weights: exactly the three tensors above. The practice file equals the
  repository's copy.
- **The editor's 86 scripts and their outputs** (the session's raw command list is not available
  to the host, as in phase 14): no internet; no training (no backward pass, autograd or optimizer);
  no generation outside the evaluation tool (the only engine-related lines import two kernel
  functions of the engine's library for a teacher-forced comparison); every path inside its own
  sandbox; the reference hash it compared the weight file with is the download record inside the
  checkpoint.
- **During passes.** Three analyses of saved activations that use only the processor were written
  while passes 1 to 3 were running (weight statistics; two covariance and threshold analyses on
  tensors already moved to the processor). No job of the editor's used the graphics card during a
  pass, as far as its scripts show; the rule forbids only graphics-card jobs during a pass.
- **Disclosures:**
  1. The editor ran through the harness's built-in general helper, which has more tools than the
     editor types; by instruction it used Read, Write, Edit and Bash only (its own statement; its
     86 scripts are shell and Python and show no other use).
  2. One host message, the word "continue", after the connection error.
  3. A byte-code cache folder (`__pycache__`, one file) appeared under the sandbox's tools folder,
     which the briefing tells the editor not to modify, when one of the editor's scripts imported
     the run-validity module from there to read its pass summaries. No tool file changed (digests
     as in the manifest).
  4. The model's family reaches the editor through the checkpoint's own files: its notes name the
     architecture once as it stands in `config.json` (`Qwen3_5ForConditionalGeneration`) and call
     it a small Qwen3 thinking model once. The task instructions do not name it. Same situation as
     phase 14 (ISSUES #33).
  5. The editor saw a folder `techcheck_evidence` in its first directory listing and states that
     it did not open it.
  6. The harness's record of the first part of the session was overwritten by the resume.
- **Checks on the numbers.** `phase15/compute_endpoints.py` applies `phase14/endpoints.py` to this
  phase's runs and adds the exploratory comparison; every re-scored file is checked for the tree's
  checker stamp, one way of reading the thinking, and figures that its run file and its re-score
  agree on. An independent recomputation from the raw files by a second implementation reproduced
  every headline number. A code review found no blocker; the script's tests caught two defects
  before it was used (the new checks pointed at the previous experiment's runs; the phase-14
  script's command-line entry computed nothing after its main function was split) — both fixed.
  Tests: `phase15/test_compute_endpoints.py` 26, `phase14/test_endpoints.py` 53 (one new); both
  suites green together in both orders (79).

## What it changes

The protocol fixed the reading of each outcome before the session. Its words for this one (P2
wrong):

> a larger blind editor, held to the original rules and practising in distribution with 15 passes,
> finds a repair of the hinted frontier editor's level. The gap left by phase 14 was a matter of the
> editor's strength, not of the hint or of the budget.

What this run adds to that:

- **The outcome is that first branch.** 265 of 500 (53.0%) under the repository's question against
  the published edit's 271 (54.2%) and stock's 47 (9.4%): REACHES IT and REPAIR met. Under the
  reference question the edit is above the published edit, 280 of 500 (56.0%) against 258 (51.6%)
  under the repository's matcher and 56.4% against 51.8% under math-verify.
- **The author's predictions P2 and P3 were wrong**: P2 said the candidate FALLS SHORT (point
  estimate 0.80) and P3 that REPAIR is not met (0.65). P1, P4 and P5 hold.
- **One session and one seed.** The registered reading stands on one run of one editor, with one
  sample per problem; the direct comparison of the two editors (P5) is exploratory for that reason.
- **"Reaches it" is the protocol's band of ±4.0 points.** The measured difference is −1.2 points
  and its interval runs from −5.0 to 2.6, so under the repository's question the two edits cannot
  be told apart. That the edit is above the published edit is a reading of the reference question
  only (+4.4 and +4.6 points, both intervals excluding zero).
- **A repair of the published edit's level, not the same repair.** The published edit removes a
  measured loop direction in layers 15 and 19 so that the loop does not form. This edit is a
  conditional loop-breaker: it lets the loop form and ends the thinking where the model is
  detected repeating itself. It is within the original authors' rule as written ("… close it early
  regardless of the problem") and was accepted by the edit checker; whether the original authors
  count it as a repair of the same kind is theirs to say. It is not the quick fix the rules forbid
  (edits to the embedding or output layer and the like): all three tensors are in layer 14's MLP.
- **The budget was the smaller one.** 15 passes of 128 problems, not 50 of up to 512; the editor
  used 4 and ended the session by itself.
- **About a sixth of the problems still get no answer.** 87 of 500 (17.4%) are cut at the cap, 83
  of them after the thinking closed, because the model loops again while writing its answer; the
  editor looked for a targeted repair for this, found none that was safe, and left it alone.
- **Exploratory, no protocol word.** Under the repository's question the candidate is 10.8 points
  above the best blind quick fix of this repository's earlier phases (42.2%; interval 6.8 to 14.8)
  and not distinguishable from the published edit with our repair at full strength (+0.8, interval
  −3.0 to 4.6).

## Cost

- **Graphics card:** 1.55 GPU-hours in all (the protocol estimated 2 to 5): the technical check
  161 s, the four practice passes 1,953.9 s (806.8, 415.9, 394.3 and 336.9), the two scored runs
  1,609.1 s and 1,866.0 s.
- **Model plan:** about 77 million tokens are on record for the resumed part of the editor's
  session; the first part's use is not known.
- **The host's time:** about 6 hours from the plan to the scored result (2026-11-26 22:20 to
  2026-11-27 03:45), plus the write-up.

## Deviations and limits

- **The editor did not run as registered.** The registered `glm54-editor` type could not be loaded
  into the chat that runs the session, so the editor ran through the harness's built-in general
  helper, which has more tools than the editor types and is handed the user's general working
  rules at its start (they say nothing about this project, the fault or mathematics, as a probe
  before the session showed). The tool restriction was enforced by instruction: by the editor's own
  statement it used Read, Write, Edit and Bash only, and its scripts show no other use. The
  session's raw command list is not available to the host.
- **One interruption and one host message.** A connection failure on the host stopped the session
  for about 12 minutes between 00:32 and 00:44 on 2026-11-27; the host then sent the word
  "continue" and nothing else, and the editor resumed with its context. Its report and notes do
  not mention the interruption.
- **The first part of the session has no harness record.** The resume overwrote it, so the
  session's tool calls and tokens are known for the resumed part only, and the audit of what the
  editor ran rests on its scripts and their outputs.
- **"Reaches it" is a band of ±4.0 points.** The measured difference is −1.2 points and its
  interval runs from −5.0 to 2.6.
- **The mechanism is not the published edit's.** A conditional loop-breaker, inside the original
  authors' rule as written and accepted by the edit checker; whether the original authors count it
  as a repair of the same kind is theirs to say.
- **About a sixth of the problems get no answer** (87 of 500, 17.4%, cut at the cap).
- **The model's family reaches the editor through the checkpoint's own files.** The protocol says
  the editor is not told its name, and the task instructions do not name it; the checkpoint's
  `config.json` carries the architecture, and the editor's notes repeat it once.
- **A byte-code cache folder appeared under the sandbox's tools folder** (one file; no tool file
  changed).
- **The editor's diagnosis is its own reading**, measured with its own scripts. The audit after
  the session checked the tools, the model, the edit and what the scripts do not do, as listed
  above.
- **Why the editor kept the pass-3 state is its own account**: a rule it fixed before pass 4, that
  the challenger had to be clearly better.
- One session, one seed, one sample per problem.

## Files added for this phase

- `phase15/results.md` — this file.
- `phase15/protocol.md` — pre-registered, with its notes written before the session;
  `phase15/sandbox_manifest.json` — the sandbox as built before the session.
- `phase15/endpoints.json` — every number of the two endpoints and of the exploratory comparisons
  (against the phase-14 editor's final and four other versions of the model).
  `phase15/compute_endpoints.py` computes it, by way of `phase14/endpoints.py`; its tests are
  `phase15/test_compute_endpoints.py` (26 tests).
- `phase15/session/` — the ledger, four pass summaries (`eval_pass_1.json` to `eval_pass_4.json`),
  four generation files (`gen_1.jsonl.gz` to `gen_4.jsonl.gz`), three receipts (`receipts/`), the
  record of host messages (`host_messages.md`), and `work/`: the editor's `final_report.md`,
  `notes.md`, its four pass outputs, 86 scripts under `work/scripts/`, 45 outputs under
  `work/out/`.
- `phase3c/p15_final*` and `phase3c/p15_final_ref*` — run, generations, re-score. Three logs under
  `phase3c/logs/`: `p15_final.log`, `p15_final_ref.log`, `grade_reference_p15.log`.
- `phase14/endpoints.py` changed: its `main()` was split into `build()` (returns the endpoints,
  writes nothing) and `main()` (prints and writes); `phase14/test_endpoints.py` gained one test of
  `main()` (53 tests).
- On the public page (`phase7/timeline_template.html`, built by `phase7/build_timeline.py`): the
  edit is the twentieth version.
