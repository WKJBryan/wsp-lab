# Phase 16 results — two more sessions of the GLM-5.4 editor: the blind repeat scores 170 of 500 (34.0%) and the told session 199 of 500 (39.8%); by the registered rule the hint's effect is INCONCLUSIVE

**Date:** 2026-11-27 (local time) · **Protocol:** `phase16/protocol.md` (pre-registered
2026-11-27, before either editor session and before any practice pass of this phase; its two notes
of technical checks were written before the sessions and changed no rule, endpoint or prediction) ·
**Model:** Qwen3.5-0.8B · Editors that ran: GLM-5.4 in two sessions, both through the harness's
built-in general helper, as the protocol registers for this phase (session A blind, 15 passes
available, 4 used; session B told, 15 passes available, 11 used). The token cap of one generation
is 32,768 tokens ("the cap").

> **What was registered, and how the run deviated.** The protocol fixed, before any session: two
> sessions of the GLM-5.4 editor under the original researchers' rules (enforced by the edit
> checker), with the same rules, practice set and budget as phases 14 and 15 — session A blind,
> identical to phase 15, and session B told, with the same briefing plus the three sentences that
> name the fault (the model's thinking loops; below, "the hint"), in the original researchers'
> wording. For each session it fixed the endpoints: the finals on MATH-500 under the repository's
> question, paired against stock and the published edit; the reference question; and, exploratory
> and without a protocol word, the comparisons with the first session's final and the phase-14
> editor's final. It fixed the words of the verdict, the rule for the hint and ten predictions;
> everything else is inherited from `phase14/protocol.md` and `phase15/protocol.md`. Written before
> each session, it also records a four-problem technical check in the fresh sandbox (2026-11-27,
> 11:46 and 16:36; its outcome sets nothing). The run deviated as follows, disclosed here, in
> ISSUES #39–#46 and under audit below; three of the items are corrections by the host.
> **The editor's helper:** both sessions ran through the harness's built-in general helper, as the
> protocol registers for this phase (in phase 15 that was a deviation, ISSUES #35: the registered
> helper type cannot be loaded into a chat that is already running).
> **Corrected, where the audit looked for scripts:** the host's audit and collection scripts looked
> for an editor's scripts in one folder (`work/scripts/`, where the editors of phases 14 and 15 had
> put them), the blind repeat's editor used `work/analysis/` and the told session's the top of its
> work folder, so the first audit of session A read no script and reported nothing found and the
> first collection took none; both scripts were corrected to read every script anywhere under the
> work folder and re-run, and the audit findings of both sessions are from the corrected scripts.
> **Corrected, what was written during passes:** the host's first check read the wrong column of
> the ledger for the end of a pass and looked at an empty window, so it reported that nothing was
> written during a pass, and that sentence stands in the commit of session A's evidence; the
> corrected check shows that both editors kept doing analyses during their passes, all on the
> processor as far as their files show, with no file of a graphics-card job of theirs dating from
> any pass window. **One attempt stopped and refunded (session B):** the
> sixth attempt was stopped by the editor three seconds after it was charged — it had been
> launched with an empty hypothesis, by the editor's own mistake — and was refunded by the tool,
> and nothing was measured in it. **A byte-code cache folder (session B):** `__pycache__`, four
> files, appeared under the sandbox's tools folder when the editor's scripts imported three
> modules from there; no tool file changed.
> **"Blind" is relative to the briefing:** the checkpoint's own model card names the fault
> (thinking loops), and the blind repeat's editor cites that sentence as its lead; on record since
> phase 3 (ISSUES #8).
> **The rule for the hint:** it cannot return NO MEASURABLE EFFECT once the two blind finals are
> 19.0 points apart (it asks for the told final to be within 4.0 points of both), so prediction B3
> (point estimate 0.55) could not hold after session A was scored, whatever session B scored; the
> author wrote the rule expecting the two blind finals to be close.
> **The scope of the stop repair (session B):** the final edit has a part (the stop repair) that
> changes when the thinking ends, the editor left the scope decision to the host, the host scored
> the weights as the editor left them, as the protocol registers, and did not score the repetition
> repair alone, and whether that part is in the original researchers' scope is theirs to say.
> **The cost estimate:** the protocol's was low.
> **The order of audit and scored runs (session B):** the scored runs were started after the
> sandbox part of its audit (the tools, the base model, the receipt, the edit re-derived from the
> weights) and while the rule audit of its scripts was being completed; that audit was finished
> before the scored runs were collected. **Two questions left unanswered:** both editors' reports
> end with a question to the host (session A: whether to force the answer phase to stop, which the
> briefing rules out; session B: the scope of the stop repair), and neither was answered.
> **Corrected, the repository-wide parse test:** after the phase-16 evidence was collected,
> `test_parse_all.py` failed on three of the editors' own scripts (two of session A's use a syntax
> allowed only from Python 3.12, and one of session B's, `val_E.py`, uses a keyword as an attribute
> name); the scripts are evidence and stay exactly as written, the test now leaves out every folder
> whose name starts with `session` and parses the repository's own code as before, and the whole
> suite was green after the change in the repository's, the grader's and the evaluation tool's
> environments (301 tests, 148 endpoint tests in both orders, 63 tests of the phase-14 tools).

## The verdict

Phase 16 ran the GLM-5.4 editor twice more under the original researchers' rules, with the practice
set and the budget of the first session (phase 15): session A, the blind repeat, briefed exactly as
the first session was; session B, the told session, briefed with the three sentences that name the
fault. The told session found a two-part repair: it inverts the self-reinforcing vote of five
attention heads (small units of the model's attention) for the repeated word "Wait", and adds one
detector neuron that ends the thinking (the model's working-out, which it must end before it
writes its answer) where a line states a result. Its edit scores **199 of 500 (39.8%)** on MATH-500
(the scored set: 500 problems, one sample each) under the repository's question (the question this
repository asks the model, read by the repository's answer checker): far above stock, the unedited
model (47 of 500, 9.4%), below the published edit, the original researchers' edit (271 of 500,
54.2%), and **between the two blind sessions of the same editor** — 13.2 points below the first
(265 of 500, 53.0%) and 5.8 points above the blind repeat (170 of 500, 34.0%). Under the reference
question (the original researchers' kind of question; the last boxed answer) it gets 187 of 500
(37.4%), against 258 of 500 (51.6%) for the published edit, 280 of 500 (56.0%) for the first blind
session and 135 of 500 (27.0%) for the blind repeat. The editor used 11 of its 15 practice passes.

In the protocol's words: against stock, **REPAIR is met** — the rule needs at least 80 of 500 right
and at least stock + 10.0 points (50 of 500); the told session's edit has 199 right and 152 over
stock, a difference of +30.4 points (95% interval 26.4 to 34.6). Against the published edit,
**FALLS SHORT** — the protocol's word for more than 4.0 points below it, with the interval
excluding zero; the difference is −14.4 points (95% interval −18.4 to −10.6).

And the phase's own question — does the hint help? — by the rule registered before the sessions:
**INCONCLUSIVE**. Against the first blind session the told final is −13.2 points (95% interval
−17.0 to −9.4); against the blind repeat it is +5.8 points (95% interval 2.2 to 9.4). The rule
gives THE HINT HELPS for more than +4.0 points above each blind final, THE HINT HURTS for more
than 4.0 points below each, and NO MEASURABLE EFFECT for within ±4.0 points of each; the told
final is more than 4 points below one blind final and more than 4 points above the other, so it is
none of the three.

(A point is a percentage point of the 500-problem score. Each difference is paired problem by
problem on the same 500 problems; its 95% interval comes from a 10,000-draw bootstrap and its p
value from an exact McNemar test, as in phase 8 and as set out under session A's finals. The host
is the agent that runs the editor sessions; the harness is the software a session runs in. Every
score on the scored set in this file is the final answer checker's reading, stamp `#d74fdebd340f`,
ISSUES #19, #30, #31; so is every comparator's.)

So the outcome has two parts.

- **On the hint:** the told session lands inside the range the same editor covers blind. Nothing in
  this phase shows that being told lifts the score above what a blind session reaches. One told
  session, set against two blind sessions 19.0 points apart, cannot show a smaller effect either
  way. What the hint went along with, in this one session, is the editor's route: it went for what
  drives the repetition first — its first edited pass came 72 minutes after the start, against 80
  and 123 in the blind sessions — and its final edit attacks the repetition at its source. Both
  blind sessions instead strengthened the model's own end-of-thinking signal: the first session
  where repetition is detected, the repeat at every line end.
- **On the first session's result:** the branch "Session A far from 265 of 500" of the protocol's
  "What each outcome would change": one session of even a strong editor is not a measurement.

**What stands after three sessions of this editor** (from the table under "The closing table"):

- The same editor, under the same rules, gave three different edits in three sessions: a
  conditional loop-breaker in the first, a closing gain in the blind repeat, and in the told
  session two repairs, the repetition repair and the stop repair.
- Their scores under the repository's question are 265 of 500 (53.0%), 170 of 500 (34.0%) and 199
  of 500 (39.8%).
- The two blind sessions are 19.0 points apart (95% interval −23.0 to −15.0); the told session
  lies between them.
- The first session's edit is the only one of the three that ends the thinking on nearly every
  problem (496 of 500); the other two leave most problems cut at the cap (cut off at the token
  limit before the answer is finished).
- Of the 500 problems, 137 are right under all three edits and 289 under at least one.

**Caveats that must stand next to the headline:**

- **One sample per problem for every scored run** (each problem is answered once); one told
  session against two blind ones, **each with one seed** (the number that fixes a run's chance
  events).
- **The three sessions end with three different edits** (under "The edit" of each session).
- **The told session's edit has a part (the stop repair) that changes when the thinking ends;**
  whether it is in the original researchers' scope is theirs to say.
- **"Blind" is relative to the briefing:** the checkpoint's own model card names the fault.
- **The edits of the two later sessions leave most problems without a finished answer:** 328 of
  500 (65.6%) for the blind repeat and 307 of 500 (61.4%) for the told session are cut at the cap.
- **The told session was the long one** (5 h 57 min, 11 passes)
  **and the first blind one the short one that scored highest** (about 3 h 40 min, 4 passes).

## The closing table

The three sessions of the same editor, side by side. All measured. The scores are on MATH-500, one
sample per problem.

| | The first session (phase 15) | The blind repeat (session A) | The told session (session B) |
|---|---|---|---|
| Briefing | blind | blind, the same file byte for byte | told: the same briefing with three sentences that name the fault |
| Score, the repository's question | 265 of 500 (53.0%) | 170 of 500 (34.0%) | 199 of 500 (39.8%) |
| Score, the reference question (repository matcher) | 280 of 500 (56.0%) | 135 of 500 (27.0%) | 187 of 500 (37.4%) |
| Against stock (47 of 500, 9.4%) | +43.6 points (95% interval 39.2 to 48.0) | +24.6 points (20.8 to 28.6) | +30.4 points (26.4 to 34.6) |
| Against the published edit (271 of 500, 54.2%) | −1.2 points (−5.0 to 2.6): REACHES IT | −20.2 points (−24.2 to −16.2): FALLS SHORT | −14.4 points (−18.4 to −10.6): FALLS SHORT |
| Passes used | 4 of 15 | 4 of 15 | 11 of 15 (one attempt stopped and refunded) |
| Time to the first edited pass | 123 minutes (with a 12-minute interruption) | 80 minutes | 72 minutes |
| Length of the session | about 3 h 40 min | 2 h 58 min | 5 h 57 min |
| Tool calls on record | 169 (the resumed part only) | 358 | 488 |
| Tokens on record | about 77 million (the resumed part only) | about 132 million | about 140 million |
| The edit | a conditional loop-breaker: two switches in two neurons of layer 14 end the thinking where repetition or a runaway list is detected | a closing gain: the output of 69 neurons of layer 23 toward the end-of-thinking token, × 2.1 | five copy heads' vote for the word "Wait" inverted (layers 15, 19, 23), plus a stop detector neuron in layer 23 |
| Tensors changed, values | 3 tensors, 6,143 values | 1 tensor, 68,246 values | 6 tensors, 1,178,485 values |
| Thinking closes (scored set) | 496 of 500 (99.2%) | 331 of 500 (66.2%) | 226 of 500 (45.2%) |
| The answer finishes (scored set) | 413 of 500 (82.6%) | 172 of 500 (34.4%) | 193 of 500 (38.6%) |
| Host messages | one ("continue"), after a connection failure | none | none |
| Graphics-card time | 1.55 GPU-hours | 2.45 GPU-hours | 3.00 GPU-hours |

Plain reading of the table, and no more than it shows: the same editor, under the same rules, gave
three different edits in three sessions; their scores are 265 of 500 (53.0%), 170 of 500 (34.0%)
and 199 of 500 (39.8%). The two blind sessions are 19.0 points apart; the told session lies between
them. The first session's edit is the only one of the three that ends the thinking on nearly every
problem (496 of 500); the other two leave most problems cut at the cap. Of the 500 problems, 137
are right under all three edits and 289 under at least one.

## Session A — the blind repeat

### The outcome

Run again the same way (the same briefing file as the first session, byte for byte, and nothing
said about the fault), the same blind editor found a **different and smaller** repair. Its edit
gets **170 of 500 (34.0%)** right on MATH-500 under the repository's question: well above stock
(47 of 500, 9.4%), and well below both the published edit (271 of 500, 54.2%) and the same editor's
first session (265 of 500, 53.0%). Under the reference question it gets 135 of 500 (27.0%), against
258 of 500 (51.6%) for the published edit and 280 of 500 (56.0%) for the first session. The editor
used 4 of its 15 practice passes.

In the protocol's words: against stock, **REPAIR is met** — the edit has 170 right and 123 over
stock, a difference of +24.6 points (95% interval 20.8 to 28.6). Against the published edit,
**FALLS SHORT** — the difference is −20.2 points (95% interval −24.2 to −16.2).

So the outcome is the third branch of the protocol's "What each outcome would change", "Session A
far from 265 of 500": one session of even a strong editor is not a measurement. Phase 15's result
was one draw, and the distance between the two blind sessions is the first estimate of how much
this editor varies from session to session.

### The finals on MATH-500

500 problems, one sample each, the repository's seed, run only after the edit checker had accepted
the final weights. Both scored runs pass both validity checks (fragment tails 0.0%; no dependence
on the request's position in the batch). All numbers are in `phase16/endpoints_A.json`, computed by
`phase16/compute_endpoints.py`, which applies `phase14/endpoints.py` to this phase's runs; the runs
are `phase3c/p16_A_final*` and `phase3c/p16_A_final_ref*` (run, generations, re-score), and the
reference grader's output is `phase3c/logs/grade_reference_p16A.log`. Every re-scored file carries
the stamp `phase3c/rescore.py+phase3b/scorer.py#d74fdebd340f (ISSUES #19, #30, #31)`, so every
comparator is read by the same final answer checker.

Paired statistics as in phase 8: each comparison is made problem by problem on the same 500
problems; the interval comes from a 10,000-draw bootstrap with seed 0, the problems in sorted order
(the 500 problems are re-drawn at random that many times, to see how much the difference moves); p
comes from an exact McNemar test (it looks only at the problems that just one of the two runs gets
right, and gives the chance of so lopsided a split if neither run were better).

**Primary — the repository's question**

The model works out loud in a thinking section and must end it before it writes its answer:
"closes its thinking" counts the generations that end it, and "cut at the cap" those that ran into
the token limit before they finished. "Mean tokens" is the mean length of a generation; "card time"
is the graphics card's time for the run.

| Run | Right | Closes its thinking | Cut at the cap | Mean tokens | Card time | Validity |
|---|---|---|---|---|---|---|
| Session A's final edit | **170 of 500 (34.0%)** | 331 of 500 (66.2%) | 328 of 500 (65.6%) | 23,484 | 2,874.7 s | both checks passed, no problem |
| Stock (the unedited model) | 47 of 500 (9.4%) | 49 of 500 (9.8%) | 453 of 500 (90.6%) | 30,108 | 3,016.2 s | passed |
| Published edit | 271 of 500 (54.2%) | 418 of 500 (83.6%) | 132 of 500 (26.4%) | 20,645 | 1,839.0 s | passed |
| The same editor's first session (phase 15) | 265 of 500 (53.0%) | 496 of 500 (99.2%) | 87 of 500 (17.4%) | 16,308 | 1,609.1 s | passed |

| Comparison | Difference | 95% interval | Right only under this edit / only under the other | p | The protocol's word |
|---|---|---|---|---|---|
| Session A's edit against stock | +24.6 points | 20.8 to 28.6 | 126 / 3 | 1.05e-33 | **REPAIR met**: it needs at least 80 of 500 right and at least stock + 10.0 points (50 of 500); the edit has 170 right and 123 over stock |
| Session A's edit against the published edit | −20.2 points | −24.2 to −16.2 | 10 / 111 | 1.04e-22 | **FALLS SHORT** |
| Session A's edit against the same editor's first session (phase 15) | −19.0 points | −23.0 to −15.0 | 11 / 106 | 1.16e-20 | no protocol word (exploratory) |
| Session A's edit against the phase-14 editor's final (55 of 500, 11.0%) | +23.0 points | 19.0 to 27.2 | 126 / 11 | 6.65e-26 | no protocol word (exploratory) |

- 44 problems are right under both this edit and stock; 3 only under stock.
- 160 problems are right under both this edit and the published edit; 10 only under this edit; 111
  only under the published edit.
- 159 problems are right under both blind sessions of this editor; 11 only under session A's edit;
  106 only under the first session's.
- How the 500 generations of this edit end: the thinking closes on 331 and never on 169. 172 finish
  (160 of them right); 328 are cut at the cap (10 of them right, all ten after the thinking had
  closed).
- Closing its thinking is not finishing. Of the 331 that close their thinking, 159 are then cut at
  the cap while writing the answer: on 159 of 500 problems (31.8%) the model ends its thinking and
  then never finishes its answer.
- All 170 right answers come from generations that closed their thinking.

**Secondary — the reference question** (the original researchers' kind of question; the last boxed
answer; both graders on the same boxed answers; paired against the reference runs of stock,
`ref_stock`, and of the published edit, `ref_orig`).

| Run | Repository matcher | math-verify | A boxed answer found | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|
| Session A's edit (`p16_A_final_ref`) | **135 of 500 (27.0%)** | **136 of 500 (27.2%)** | 143 of 500 (28.6%) | 71.2% (356 of 500) | 25,051 | 3,139.0 s |
| Stock (`ref_stock`) | 44 of 500 (8.8%) | 44 of 500 (8.8%) | 44 of 500 (8.8%) | 93.2% (466 of 500) | 30,991 | 2,871.9 s |
| Published edit (`ref_orig`) | 258 of 500 (51.6%) | 259 of 500 (51.8%) | 320 of 500 (64.0%) | 29.0% (145 of 500) | 21,796 | 1,788.0 s |

| Comparison | Repository matcher | math-verify |
|---|---|---|
| Session A's edit against stock | +18.2 points (95% interval 14.8 to 21.8; 94 problems right only under this edit, 3 only under stock; p = 1.92e-24) | +18.4 points (95% interval 15.0 to 22.0; 95 problems right only under this edit, 3 only under stock; p = 9.9e-25) |
| Session A's edit against the published edit | **−24.6 points (95% interval −28.6 to −20.4; 9 only under this edit, 132 only under the published edit; p = 3.6e-29)** | **−24.6 points (95% interval −28.6 to −20.4; 9 only under this edit, 132 only under the published edit; p = 3.6e-29)** |

No protocol word attaches to the reference comparisons. Read by the repository's usual rule (which
credits answers stated outside the box), the same reference run gives 143 of 500 (28.6%).

**Other exploratory comparisons of session A's edit** (the repository's question; no protocol word)

| Against | Difference | 95% interval | Right only under this edit / only under the other | p |
|---|---|---|---|---|
| The best blind quick fix (MiMo-flash, phase 4b; `p4_mimo`; 211 of 500, 42.2%) | −8.2 points | −12.0 to −4.2 | 31 / 72 | 6.6e-05 |
| The 50-pass edit (phase 10; `p10_fullbudget`; 208 of 500, 41.6%) | −7.6 points | −11.6 to −3.6 | 35 / 73 | 0.000326 |
| The published edit with our repair at full strength (phase 7; `p7_stack_full_v2`; 261 of 500, 52.2%) | −18.2 points | −22.2 to −14.2 | 14 / 105 | 2.05e-18 |
| Our map repair (phase 7; `p7_uperp_on_loop`; 182 of 500, 36.4%) | −2.4 points | −6.2 to 1.4 | 41 / 53 | 0.256 |

In words: this edit is below the best blind quick fixes of the earlier phases (which the original
researchers' rules forbid), and not distinguishable from our own map repair.

### The session

**The editor, and how it ran.** The editor was GLM-5.4, blind. It ran through the harness's built-in
general helper, as the protocol registers (the same way as phase 15). Its briefing was the editor
types' four operating rules, the sentence that its tools are exactly Read, Glob, Grep, Write, Edit
and Bash, and the path of `phase14/EDITOR_PROMPT.md` — the same briefing file as phase 15, byte for
byte. Nothing was said about the fault. The helper has more tools than that, so the restriction was
enforced by instruction and audited afterwards (below). The sandbox was fresh: its base model and
its seven tools are those of phase 15 (same digests), and the four-problem technical check had run
in it before the session, after which the ledger and counter were reset, so that it held no pass
when the session began (the protocol's note, written before the session).

**Timeline** (local time, 2026-11-27; the harness's record):

- Started 11:47:50, completed 14:46:19 — 2 hours 58 minutes.
- Pass 1, on the unedited model, ran 11:48:51 to 12:02:18.
- The first edited pass began at 13:08:14, 80 minutes after the start. (The first session: 123
  minutes, including a 12-minute interruption and its 13-minute baseline pass.)
- Passes 2 to 4 ran 13:08:14 to 13:18:33, 13:46:28 to 13:56:45 and 14:03:46 to 14:13:48.
- The final report was first written at 14:19 and revised while the editor went on analysing; the
  receipt of its final state was written at 14:45:31.
- The editor ended the session by itself.

**Length and use.** 178.5 minutes, 358 tool calls, about 132 million tokens (381 thousand of them
output; about 131 million were reads of input the model service had cached). For scale, the first
session took about 3 h 40 min; its record on the harness covers the resumed part only (114 minutes,
169 tool calls, about 77 million tokens).

**Passes.** 4 of 15 charged, none refunded, no candidate refused by the checker (two receipts).

**Scored runs.** The repository's question ran 14:50:37 to 15:40:00 and the reference question
15:40:00 to 16:33:22; the reference grader took 3 seconds.

**No host message** was sent at any time, and no connection failure occurred;
`phase16/session_A/host_messages.md` is the record. The editor's report ends with a question to
the host (below, under "Audit against the manifest and the rules", disclosure 6).

### The practice passes

A practice pass runs the 128 practice problems through the evaluation tool on one state of the
weights, and costs one of the editor's 15 passes. The numbers are the tool's summaries
(`phase16/session_A/eval_pass_1.json` to `eval_pass_4.json`); the generations are in
`gen_1.jsonl.gz` to `gen_4.jsonl.gz` and the ledger is `phase16/session_A/ledger.md`. All four
passes pass both validity checks. Under "Right, boxed rule" a problem is right when its boxed
answer is; "the repository's reading" is its usual rule, which also credits answers stated outside
the box.

| Pass | State | Right, boxed rule | Right, repository's reading | Closes its thinking | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|---|
| 1 | the unedited model | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 121 of 128 (94.5%) | 31,286 | 807.7 s |
| 2 | the closing gain at 2.1 — **the final edit** | 37 of 128 (28.9%) | 41 of 128 (32.0%) | 80 of 128 (62.5%) | 88 of 128 (68.8%) | 24,164 | 619.0 s |
| 3 | the closing gain at 2.3 | 35 of 128 (27.3%) | 42 of 128 (32.8%) | 100 of 128 (78.1%) | 86 of 128 (67.2%) | 23,223 | 617.0 s |
| 4 | the pass-2 weights again | 37 of 128 (28.9%) | 41 of 128 (32.0%) | 80 of 128 (62.5%) | 88 of 128 (68.8%) | 24,164 | 601.9 s |

- Paired on the same problems (boxed rule): pass 2 against pass 1, 30 gained and none lost; pass 3
  against pass 2, 6 gained and 8 lost.
- Pass 4's texts are identical to pass 2's, token for token (the host compared them): the evaluation
  is deterministic for fixed weights (the same weights always give the same texts), so a repeat adds
  nothing. The editor ran it to learn that.

| Level | Problems | Pass 1 (the unedited model) | Pass 2 (the final edit) | Pass 3 |
|---|---|---|---|---|
| 1 | 11 | 0 | 8 | 7 |
| 2 | 23 | 4 | 9 | 9 |
| 3 | 27 | 3 | 8 | 10 |
| 4 | 33 | 0 | 9 | 7 |
| 5 | 34 | 0 | 3 | 2 |

(Right by level, boxed rule.)

- How the practice generations end. Pass 2: of the 80 that close their thinking, 40 answers finish
  (37 right) and 40 run to the cap (none right). Pass 3: of the 100, 42 finish (35 right) and 58 run
  to the cap (none right).
- Fingerprints (identifiers of a state of the weights): pass-2 state, the final, `cd55253ac536fe76`;
  pass-3 state `cfde5d09c6f8fd48`.
- The editor kept the pass-2 state: the stronger gain closed the thinking on more problems but did
  not raise the number right.
- Its forecast for the scored set, written before any scored run: about 29 of 100 right (25 to 32),
  against about 4 to 6 of 100 for the unedited model. Measured: 170 of 500 (34.0%) under the
  repository's question, above that range, and 135 of 500 (27.0%) / 136 of 500 (27.2%) under the
  reference question, inside it.
- Why it stopped with 11 passes left (its own reasons): a repeat is uninformative (pass 4), and
  every remaining candidate it could design was estimated at +3 points or less, below what one pass
  can resolve (a pass on 128 problems has a standard error — the spread that chance alone gives —
  of about 4.4 points).

### The edit

From its receipt (`phase16/session_A/receipts/`, written at 14:45:31); re-derived from the weights
by the host after the session, who found exactly this.

- One tensor (one of the model's weight tables):
  `model.language_model.layers.23.mlp.down_proj.weight`, the output matrix of the feed-forward
  block of layer 23, the last layer; shape 1024 × 3584.
- 68,246 of its 3,670,016 values changed; the largest change 0.007153; the size of the change (its
  norm) 0.159488, against 16.42 for the whole matrix.
- The other 487 tensors and all 28 non-weight files are identical to the base model.

**How it works (the editor's description).** 69 neurons of that block — columns 2, 7, 18 and 66
others up to 2220 — fire at the end of a line inside the thinking and almost nowhere else, and they
write toward the model's end-of-thinking token (the mark that ends the working-out and starts the
answer). For each of those 69 columns, the part of its output vector that points along the read-out
direction of that token (the direction along which the model's final state is scored for it) is
multiplied by 2.1, in float32 and rounded back to bfloat16 (the arithmetic in a more precise number
format, the result stored in the model's own); every other direction of every neuron is unchanged.
Measured effect: the model's score for that token (its logit, in nats) rises by about 7 at line ends
inside the thinking, by about 2.1 in the answer, 1.9 in the middle of a line and 0.5 at the first
generated token. Away from line ends, the probability the model assigns to the tokens it actually
wrote changes by −0.0005 on average.

**The rule, and the mechanism.** The briefing puts "making the model skip its thinking or close it
early regardless of the problem" out of scope. The editor's argument that its edit is inside the
rule: the neurons, the layer and the direction were located by causal attribution (by measuring
which parts of the model cause that score); the gain acts on an existing pathway and only where it
fires (line ends); and the outcome depends on content, because the model's own weak closing signal
decides at which line ends the closing wins — at 21 ordinary line ends of the baseline generations
the closing probability is 0.034 on average at gain 2.1, and would be 0.467 at gain 3, which would
close regardless of content. Against that stands the editor's own simulation: at gain 2.1 the
thinking would end before the answer appears on 19.5% of the baseline problems (4.6% before token
400). The edit checker accepted the edit (it changes no embedding or output layer).

**Kind, by the host's reading** of the receipt and the report — the protocol's third kind, "other":
**a closing gain**. It is not the published edit's mechanism — that one removes a measured loop
direction in layers 15 and 19 so that the loop does not form — and not the first session's — that
one is a conditional loop-breaker that ends the thinking where repetition is detected. This edit
neither detects repetition nor touches what drives it; it makes the model's own "end the thinking"
signal about twice as strong at every line end. In effect it is a milder relative of the quick fix
the original researchers' rules forbid (which raises the same token from the output layer). Whether
the original researchers count it as a repair is theirs to say.

**A candidate the editor built and did not submit:** the closing gain plus the output columns of
four loop-sustaining heads multiplied by 0.35 (four tensors). In real forward passes it lowered the
probability of the tokens the model actually writes by 0.51 nats on average (the closing gain alone:
0.0005), so the editor rejected it before a pass.

### The editor's diagnosis

The editor's own report is `phase16/session_A/work/final_report.md`, its journal
`phase16/session_A/work/notes.md` (it headed both as phase 14: the briefing file it was given is
phase 14's). What follows is its reading, from forward passes on fixed texts (the model is run over
texts already written; nothing is generated), which cost no pass; the measurements are the editor's.

- **The failure, in its reading.** 121 of 128 baseline generations are cut at the cap; 118 of them
  end in a verbatim loop (one sentence of a median 37 tokens, repeated hundreds to thousands of
  times), which begins after a median of about 7,200 tokens. In 66 of the 121 the correct answer
  appears as plain text within the first 2,000 tokens.
- **The decision to end the thinking is taken at one place:** the token after a line break. At
  ordinary line ends the token that continues the model's format leads the end-of-thinking token by
  about 14.5 nats; after "So the answer is X." by about 8; only after lines like "I'll write the
  response now" by about 2.5. The margin does not shrink with time: the model has no "enough
  thinking" signal.
- **Where the closing signal is written.** It is written almost entirely in the last layers; the
  largest single writer is the feed-forward block of layer 23 (about +7 nats at every line end),
  through the 69 neurons; layers 0 to 13 contribute about +1.2 to +1.6 in total.
- **The model barely encodes "the answer has been found":** a linear classifier (the simplest kind
  of divider between two groups of examples) separating line ends that state a conclusion from
  other line ends reaches 61% to 62% at every layer.
- **Why the loop has no local repair.** The loops are sustained by a few early heads (above all one
  head of layer 0: removing it at a loop position lowers the loop token by 9.5 nats). But the same
  change written into the weights damages normal prediction, because those heads carry the context
  for it. So the loop has a local cause and no local repair that leaves the model intact.
- **Not repaired: the answer.** After the thinking ends, the answer often fails the same way — the
  model must choose between starting its final boxed line and writing more, and in answers that
  never finish the boxed token stays 13.6 nats below the top token whether or not the answer was
  already stated. The editor looked for a targeted repair, found none that was clean, and did not
  force the model to stop.
- **A correction it made to its own work:** one of its analysis tools had a defect (a wrong
  attention mask whenever a continuation started after layer 0). It caught the defect with a
  control, fixed the tool and redid the two analyses that depended on it. The edit, its verification
  and the four passes do not involve that tool.
- **It did not check whether the checkpoint equals the public release** (no internet, no reference
  copy).
- **Its lead came from the model card.** The blind briefing does not name the fault, but the
  checkpoint's own `README.md` (line 856) does: in thinking mode this model "is more prone to
  entering thinking loops … which may prevent it from terminating generation properly". The editor's
  notes cite that sentence as their lead. The remark has been on record in this repository since
  phase 3 (ISSUES #8). So "blind" means here: not told by the briefing; the fault is named in a file
  the editor may read. The same file was in the sandbox of phases 14 and 15.
- **Its forecast used the level mix of MATH-500 "from memory"**, as its report says; it could not
  check it without internet.

### Audit against the manifest and the rules

Done by the host after the session.

- **Tools, model, edit.** The seven tool files carry the digests of `phase16/sandbox_manifest_A.json`;
  the base model's fingerprint equals the manifest's; the final weights (fingerprint
  `cd55253ac536fe76`) carry their receipt; two receipts in all, no refusal. The edit was re-derived
  from the weights: exactly the one tensor above, 68,246 values. The working copy's files have the
  same names as the base's.
- **The editor's 50 scripts (47 Python, 3 shell) and their outputs**, all in one folder of its own,
  `work/analysis/`: no internet; no training (no backward pass, autograd or optimizer); no generation
  outside the evaluation tool (its forward passes are the teacher-forced reads — runs over fixed
  texts — that the briefing allows without limit; no script grows a text step by step); every path
  inside its own sandbox; no script imports from the tools folder; no byte-code cache under the
  tools folder.
- **During passes.** The editor's own analyses went on while its passes ran, which the briefing allows
  as long as they do not use the graphics card. During pass 1 it wrote and ran weight statistics and a
  tokenizer alignment, both on the processor, and saved a print-out of library code; during each of
  passes 2 to 4 it wrote one analysis script (a per-problem summary of the pass before; an outcome
  analysis of saved generations; a validation script that uses the card, for which no output file
  dates from that pass). No file in its activation cache — which only a job that runs the model on the
  card writes — dates from any pass window; the first were written two minutes after pass 1 ended.
  The session's raw command list is not available to the host, as in phases 14 and 15, so this is
  what its files show and no more.
- **Two corrections by the host during the audit.**
  1. The audit and collection scripts of phases 14 and 15 look for an editor's scripts under
     `work/scripts/`, where the editors of phases 14 and 15 put them. This editor put them under
     `work/analysis/`. The first audit pass therefore read no script and reported nothing found; the
     first collection took none. The host noticed it from the editor's report (which names the
     folder), corrected both scripts, re-ran the audit over all 50 scripts and collected them, before
     the scored runs were collected into the repository.
  2. The host's first check of files written during passes read the wrong column of the ledger for
     the end of a pass, so it looked at an empty window and reported that nothing was written during
     a pass; that sentence stands in the host's commit of this session's evidence. The check was
     corrected and re-run on both sessions; the corrected finding is the point above.

  Nothing in the session depended on either.
- **Disclosures:**
  1. The editor ran through the harness's built-in general helper, which has more tools than the
     editor types; by instruction it used Read, Glob, Grep, Write, Edit and Bash only (its own
     statement; its 50 scripts are shell and Python and show no other use).
  2. No host message and no connection failure.
  3. The model's family and the fault's kind reach the editor through the checkpoint's own files (the
     architecture in `config.json`; the remark on thinking loops in the model card). Same situation as
     phases 14 and 15 (ISSUES #8, #33, #38).
  4. A folder of the host's four-problem technical check (`techcheck_evidence`, written before the
     session) sits in the sandbox tree. The editor's files do not mention it; the session's raw
     command list is not available to the host, so whether it listed the folder is not known.
  5. The scored runs of this session were collected and re-scored before session B began (the
     protocol's order), and session B's sandbox was built only after they had ended.
  6. The editor's report ends with a question to the host — whether the remaining failure in the
     answer phase should be attacked with a content-independent "answer budget" (which the briefing
     rules out). Nobody answered; nothing was done about it.
- **Checks on the numbers.** `phase16/compute_endpoints.py` applies `phase14/endpoints.py` to this
  phase's runs; every re-scored file is checked for the tree's checker stamp, one way of reading the
  thinking, figures that its run file and its re-score agree on, and the right kind of run (the final
  and its reference run are the same model, each under its own question). A dry run on the published
  phase-15 runs reproduces every number of `phase15/endpoints.json`. A code review and the tests
  found nine weaknesses before the script's first use; all are fixed. Tests:
  `phase16/test_compute_endpoints.py` 69; with the endpoint suites of phases 14 and 15, 148 green in
  both orders. An independent recomputation from the re-scored files and the saved generations by a
  second implementation (plain counts, scipy's exact test, a bootstrap of its own with another seed
  and 20,000 draws, its own reader of the last boxed answer) agreed with every figure above: every
  count, difference and p-value exactly, every interval within half a point.

### Cost of session A

- **Graphics card:** 2.45 GPU-hours in all — the technical check 169 s, the four practice passes
  2,645.6 s, the two scored runs 2,874.7 s and 3,139.0 s. The first session took 1.55. The protocol
  estimated about 3 GPU-hours for both sessions together; that estimate was low because this edit
  leaves 65.6% of the scored generations running to the cap (328 of 500; the first session's edit:
  87 of 500, 17.4%), which makes each scored run about 1.7 to 1.8 times as long (48 and 52 minutes
  against 27 and 31).
- **Model plan:** about 132 million tokens for the whole session.
- **The host's time:** about 5 hours from the session's start to its scored result (11:48 to 16:33),
  of which 1 h 43 min were the scored runs, plus the write-up.

## Session B — the told session

### The outcome

The told session — the same editor, briefed with the three sentences that name the fault — found a
two-part repair: it inverts five attention heads' self-reinforcing vote for the repeated word
"Wait", and adds one detector neuron that ends the thinking where a line states a result. Its edit
gets **199 of 500 (39.8%)** on MATH-500 under the repository's question: far above stock (47 of
500, 9.4%), below the published edit (271 of 500, 54.2%), and
**between the two blind sessions of the same editor** — 13.2 points below the first (265 of 500,
53.0%) and 5.8 points above the blind repeat (170 of 500, 34.0%). Under the reference question it
gets 187 of 500 (37.4%), against 258 of 500 (51.6%) for the published edit, 280 of 500 (56.0%) for
the first blind session and 135 of 500 (27.0%) for the blind repeat. The editor used 11 of its 15
practice passes.

The protocol's words: against stock, **REPAIR is met** — the edit has 199 right and 152 over stock,
a difference of +30.4 points (95% interval 26.4 to 34.6). Against the published edit,
**FALLS SHORT** — the difference is −14.4 points (95% interval −18.4 to −10.6). And the phase's own
question — does the hint help? — by the rule registered before the sessions: **INCONCLUSIVE**:
against the first blind session the difference is −13.2 points (95% interval −17.0 to −9.4),
against the blind repeat +5.8 points (95% interval 2.2 to 9.4). The rule gives THE HINT HELPS for
more than +4.0 points above each blind final, THE HINT HURTS for more than 4.0 points below each,
and NO MEASURABLE EFFECT for within ±4.0 points of each; the told final is more than 4 points below
one blind final and more than 4 points above the other, so it is none of the three.

### The finals on MATH-500

500 problems, one sample each, the repository's seed, run only after the edit checker had accepted
the final weights. Both scored runs pass both validity checks (fragment tails 0.0%; no dependence
on the request's position in the batch). All numbers are in `phase16/endpoints_B.json`, computed
as for session A; the runs are `phase3c/p16_B_final*` and `phase3c/p16_B_final_ref*` (run,
generations, re-score), and the reference grader's output is
`phase3c/logs/grade_reference_p16B.log`. Every re-scored file carries the stamp
`phase3c/rescore.py+phase3b/scorer.py#d74fdebd340f (ISSUES #19, #30, #31)`. Paired statistics as in
phase 8, as set out under session A: 10,000-draw bootstrap, seed 0, problems in sorted order; exact
McNemar test.

**Primary — the repository's question**

| Run | Right | Closes its thinking | Cut at the cap | Mean tokens | Card time | Validity |
|---|---|---|---|---|---|---|
| Session B's final edit | **199 of 500 (39.8%)** | 226 of 500 (45.2%) | 307 of 500 (61.4%) | 22,313 | 1,954.5 s | both checks passed, no problem |
| Stock (the unedited model) | 47 of 500 (9.4%) | 49 of 500 (9.8%) | 453 of 500 (90.6%) | 30,108 | 3,016.2 s | passed |
| Published edit | 271 of 500 (54.2%) | 418 of 500 (83.6%) | 132 of 500 (26.4%) | 20,645 | 1,839.0 s | passed |
| The same editor's first session, blind (phase 15) | 265 of 500 (53.0%) | 496 of 500 (99.2%) | 87 of 500 (17.4%) | 16,308 | 1,609.1 s | passed |
| The blind repeat (session A) | 170 of 500 (34.0%) | 331 of 500 (66.2%) | 328 of 500 (65.6%) | 23,484 | 2,874.7 s | passed |

| Comparison | Difference | 95% interval | Right only under this edit / only under the other | p | The protocol's word |
|---|---|---|---|---|---|
| Session B's edit against stock | +30.4 points | 26.4 to 34.6 | 153 / 1 | 1.36e-44 | **REPAIR met**: it needs at least 80 of 500 right and at least stock + 10.0 points (50 of 500); the edit has 199 right and 152 over stock |
| Session B's edit against the published edit | −14.4 points | −18.4 to −10.6 | 21 / 93 | 5.31e-12 | **FALLS SHORT** |
| Session B's edit against the first blind session (phase 15) | −13.2 points | −17.0 to −9.4 | 17 / 83 | 1.31e-11 | by the rule for the hint, against that blind final alone the word would be "THE HINT HURTS" |
| Session B's edit against the blind repeat (session A) | +5.8 points | 2.2 to 9.4 | 58 / 29 | 0.00248 | against that blind final alone the word would be "THE HINT HELPS" |
| Session B's edit against the phase-14 editor's final (55 of 500, 11.0%) | +28.8 points | 24.6 to 33.2 | 154 / 10 | 2.68e-34 | no protocol word (exploratory) |

- **The rule for the hint, as registered:** THE HINT HELPS if more than +4.0 points above each blind
  final with both 95% intervals excluding 0; THE HINT HURTS if more than 4.0 points below each with
  both intervals excluding 0; NO MEASURABLE EFFECT if within ±4.0 points of each; otherwise
  INCONCLUSIVE. Applied: **INCONCLUSIVE** — the told final lies between the two blind finals
  (−13.2 points, 95% interval −17.0 to −9.4, against the first session; +5.8 points, 95% interval
  2.2 to 9.4, against the blind repeat).
- 46 problems are right under both this edit and stock; 1 only under stock.
- 178 problems are right under both this edit and the published edit; 21 only under this edit; 93
  only under the published edit.
- 182 problems are right under both this edit and the first blind session's edit; 17 only under
  this edit; 83 only under the first session's.
- 141 problems are right under both this edit and the blind repeat's; 58 only under this edit; 29
  only under the blind repeat's.
- 137 problems are right under all three sessions' edits, 289 under at least one of them, and 211
  under none.
- How the 500 generations of this edit end: the thinking closes on 226 and never on 274 (54.8%).
  193 finish (183 of them right); 307 are cut at the cap (16 of them right).
- Closing its thinking is not finishing. Of the 226 that close their thinking, 33 are then cut at
  the cap while writing the answer; 16 of those 33 are right, by a box written before the cap.
- All 199 right answers come from generations that closed their thinking.

**Secondary — the reference question** (the original researchers' kind of question; the last boxed
answer; both graders on the same boxed answers; paired against `ref_stock` and `ref_orig`).

| Run | Repository matcher | math-verify | A boxed answer found | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|
| Session B's edit (`p16_B_final_ref`) | **187 of 500 (37.4%)** | **188 of 500 (37.6%)** | 194 of 500 (38.8%) | 62.6% (313 of 500) | 22,911 | 1,975.4 s |
| Stock (`ref_stock`) | 44 of 500 (8.8%) | 44 of 500 (8.8%) | 44 of 500 (8.8%) | 93.2% (466 of 500) | 30,991 | 2,871.9 s |
| Published edit (`ref_orig`) | 258 of 500 (51.6%) | 259 of 500 (51.8%) | 320 of 500 (64.0%) | 29.0% (145 of 500) | 21,796 | 1,788.0 s |

| Comparison | Repository matcher | math-verify |
|---|---|---|
| Session B's edit against stock | +28.6 points (95% interval 24.4 to 32.6; 145 problems right only under this edit, 2 only under stock; p = 1.22e-40) | +28.8 points (95% interval 24.6 to 32.8; 146 problems right only under this edit, 2 only under stock; p = 6.18e-41) |
| Session B's edit against the published edit | **−14.2 points (95% interval −18.2 to −10.2; 22 only under this edit, 93 only under the published edit; p = 1.41e-11)** | **−14.2 points (95% interval −18.4 to −10.2; 23 only under this edit, 94 only under the published edit; p = 2.23e-11)** |

No protocol word attaches to the reference comparisons. Read by the repository's usual rule (which
credits answers stated outside the box), the same reference run gives 197 of 500 (39.4%).

**Other exploratory comparisons of session B's edit** (the repository's question; no protocol word)

| Against | Difference | 95% interval | Right only under this edit / only under the other | p |
|---|---|---|---|---|
| The best blind quick fix (MiMo-flash, phase 4b; `p4_mimo`; 211 of 500, 42.2%) | −2.4 points | −6.4 to 1.6 | 44 / 56 | 0.271 |
| The 50-pass edit (phase 10; `p10_fullbudget`; 208 of 500, 41.6%) | −1.8 points | −5.8 to 2.2 | 49 / 58 | 0.439 |
| The published edit with our repair at full strength (phase 7; `p7_stack_full_v2`; 261 of 500, 52.2%) | −12.4 points | −16.2 to −8.6 | 19 / 81 | 2.7e-10 |
| Our map repair (phase 7; `p7_uperp_on_loop`; 182 of 500, 36.4%) | +3.4 points | −0.6 to 7.4 | 60 / 43 | 0.114 |

In words: this edit is not distinguishable from the best blind quick fixes of the earlier phases
(which the original researchers' rules forbid) nor from our own map repair.

### The session

**The editor, and how it ran.** The editor was GLM-5.4, told. It ran through the harness's built-in
general helper, as the protocol registers (the same way as phase 15 and session A). Its briefing
was the editor types' four operating rules, the sentence that its tools are exactly Read, Glob,
Grep, Write, Edit and Bash, and the path of `phase16/EDITOR_PROMPT.md`: the blind briefing with the
three sentences that name the fault, in the original researchers' wording, and nothing else
changed. The three sentences, as the protocol records them: "Repairing a specific failure mechanism
is in scope: for example, if the model often fails because it repeats itself and never finishes,
finding and fixing what drives that repetition counts."; "… and neither is making the model skip
its thinking or close it early regardless of the problem; repairing a specific failure inside the
thinking (such as repetition) is in scope." (the sentence "You are not told what is wrong with the
model." is dropped); and, in "How to work", a failure is "a wrong final answer, finishing without
an answer, or being cut off at the token limit (often while repeating itself in its thinking)". The
helper has more tools than the briefing allows, so the restriction was enforced by instruction and
audited afterwards (below). The sandbox was fresh, built after session A's scored runs had ended
so that the two sessions never shared the graphics card; its base model and its seven tools are
those of phase 15 (same digests), and the four-problem technical check had run in it before the
session, after which the ledger and counter were reset (the protocol's note, written before the
session).

**Timeline** (local time, 2026-11-27; the harness's record):

- Started 16:37:50, completed 22:34:55 — 5 hours 57 minutes.
- Pass 1, on the unedited model, ran 16:39:09 to 16:52:06.
- The first edited pass began at 17:50:12, 72 minutes after the start. (The blind repeat: 80
  minutes; the first session: 123 minutes, including a 12-minute interruption and its 13-minute
  baseline pass.)
- The edited passes, in order: attempt 2 ran 17:50:12 to 18:01:03; attempt 3 19:02:19 to 19:12:18;
  attempt 4 19:37:09 to 19:46:18; attempt 5 20:40:23 to 20:48:49; attempt 6 20:52:25 to 20:52:28
  (stopped, see below); attempt 7 20:53:07 to 21:02:16; attempt 8 21:04:12 to 21:15:58; attempt 9
  21:16:19 to 21:25:35; attempt 10 21:56:03 to 22:04:52; attempt 11 22:05:13 to 22:16:17; attempt
  12 22:17:24 to 22:27:43.
- The editor ran the edit checker on its final weights a last time at 22:34:31 and ended the session
  by itself with its final report.

**Length and use.** 357.1 minutes, 488 tool calls, about 140 million tokens (456 thousand of them
output; about 139 million were reads of input the model service had cached). For scale: the first
session took about 3 h 40 min (its record on the harness covers the resumed part: 114 minutes, 169
tool calls, about 77 million tokens); the blind repeat 2 h 58 min, 358 tool calls, about 132 million
tokens.

**Passes.** 11 of 15 charged (attempts 1 to 5 and 7 to 12); four passes unused. No candidate was
refused by the checker (nine receipts). Attempt 6 was charged at 20:52:25 and stopped by the editor
at 20:52:28, before any generation: it had been launched with an empty hypothesis, by the editor's
own mistake (a transient error reaching its working folder; its launcher did not check). The tool
recorded "INTERRUPTED, pass refunded" and returned the pass; its ledger keeps the row with the
empty hypothesis. Nothing was measured in that attempt.

**Scored runs.** The repository's question ran 22:38:11 to 23:12:14 and the reference question
23:12:14 to 23:46:10; the reference grader took 4 seconds.

**No host message** was sent at any time, and no connection failure occurred;
`phase16/session_B/host_messages.md` is the record. The editor's report ends with a decision it
leaves to the host (below, under "Audit against the manifest and the rules", disclosure 7).

### The practice passes

The same 128 practice problems and the tool's summaries (`phase16/session_B/eval_pass_1.json` to
`eval_pass_5.json` and `eval_pass_7.json` to `eval_pass_12.json`); the generations are in
`gen_1.jsonl.gz` to `gen_5.jsonl.gz` and `gen_7.jsonl.gz` to `gen_12.jsonl.gz`, and the ledger is
`phase16/session_B/ledger.md` (twelve attempts, one refunded). Every measured pass passes both
validity checks.

| Attempt | State | Right, boxed rule | Right, repository's reading | Closes its thinking | Cut at the cap | Mean tokens | Card time |
|---|---|---|---|---|---|---|---|
| 1 | the unedited model | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 7 of 128 (5.5%) | 121 of 128 (94.5%) | 31,286 | 776.6 s |
| 2 | the repetition repair alone | 40 of 128 (31.2%) | 40 of 128 (31.2%) | 42 of 128 (32.8%) | 87 of 128 (68.0%) | 25,995 | 651.5 s |
| 3 | both repairs — **the final edit** | 48 of 128 (37.5%) | 52 of 128 (40.6%) | 58 of 128 (45.3%) | 79 of 128 (61.7%) | 22,700 | 598.8 s |
| 4 | a wider flip of thirteen line-start words in nine heads, plus the stop repair | 44 of 128 (34.4%) | 45 of 128 (35.2%) | 54 of 128 (42.2%) | 80 of 128 (62.5%) | 22,584 | 549.7 s |
| 5 | both repairs, plus a gain unit on the model's own exit evidence | 46 of 128 (35.9%) | 50 of 128 (39.1%) | 84 of 128 (65.6%) | 71 of 128 (55.5%) | 21,351 | 506.2 s |
| 6 | the final weights | stopped after 3 seconds; refunded; nothing measured | | | | | |
| 7 | the final weights again, an exact re-run | 48 of 128 (37.5%) | 52 of 128 (40.6%) | 58 of 128 (45.3%) | 79 of 128 (61.7%) | 22,700 | 549.4 s |
| 8 | the repetition repair alone, with one unrelated head × 1.02 | 23 of 128 (18.0%) | 24 of 128 (18.8%) | 22 of 128 (17.2%) | 107 of 128 (83.6%) | 29,376 | 706.4 s |
| 9 | the final weights, with the same head × 1.02 | 44 of 128 (34.4%) | 46 of 128 (35.9%) | 55 of 128 (43.0%) | 80 of 128 (62.5%) | 22,743 | 555.9 s |
| 10 | the final weights, with another unrelated head × 0.98 | 46 of 128 (35.9%) | 46 of 128 (35.9%) | 57 of 128 (44.5%) | 77 of 128 (60.2%) | 21,794 | 529.0 s |
| 11 | the repetition repair alone, with the same second head × 0.98 | 30 of 128 (23.4%) | 30 of 128 (23.4%) | 33 of 128 (25.8%) | 96 of 128 (75.0%) | 27,867 | 664.1 s |
| 12 | the stop repair alone, on the unedited model | 33 of 128 (25.8%) | 37 of 128 (28.9%) | 39 of 128 (30.5%) | 95 of 128 (74.2%) | 25,295 | 619.1 s |

- **The same weights measured more than once.** Attempts 3, 9 and 10 are the final weights in three
  functionally identical copies (the two later ones differ in one unrelated attention head, scaled
  by 2%): 48, 44 and 46 right. Attempts 2, 8 and 11 are the repetition repair alone in three such
  copies: 40, 23 and 30. Attempt 7 re-ran the final weights exactly and gave texts identical to
  attempt 3's, token for token (the host compared them): the evaluation is deterministic for fixed
  weights.
- **Why the editor made the copies (its reason):** a copy indistinguishable by its own forward-pass
  measurements changes which words the sampler (the step that picks the next word) happens to pick,
  so the copies show how far the score of one set of weights moves on its own.
- **How the generations end.** The final weights (attempt 3): of 128, the thinking closes on 58 and
  never on 70. 49 answers finish (45 of them right); 9 close and are then cut at the cap while
  writing the answer (2 of them right, by a box written earlier); of the 70 that never close, 1 is
  right, by a box written before the loop. The unedited model (attempt 1): 7 close and finish (6
  right); of the 121 that never close, 1 is right. The repetition repair alone (attempt 2): 42
  close; 41 finish (39 right); 1 is cut after closing (none right); of the 86 that never close, 1 is
  right.
- **Paired on the same problems (boxed rule):** attempt 2 against 1, 34 gained and 1 lost; attempt 3
  against 2, 13 gained and 5 lost; attempt 3 against 12, 20 gained and 5 lost; attempt 12 against
  1, 27 gained and 1 lost; attempt 9 against 8, 28 gained and 7 lost; attempt 10 against 11, 20
  gained and 4 lost.

| Level | Problems | Unedited model (attempt 1) | The final weights, summed over its three measurements (attempts 3, 9 and 10) |
|---|---|---|---|
| 1 | 11 | 0 of 11 | 23 of 33 |
| 2 | 23 | 4 of 23 | 46 of 69 |
| 3 | 27 | 3 of 27 | 30 of 81 |
| 4 | 33 | 0 of 33 | 29 of 99 |
| 5 | 34 | 0 of 34 | 10 of 102 |

(Right by level, boxed rule.)

- **Fingerprints:** the repetition repair alone `22f8d829ffeadf7b`; the final weights
  `41d6229f334e500f`; the wider flip plus the stop repair `d683195c14f0499b`; with the gain unit
  `636758da58420a23`; the final weights with the first perturbed head `931c8887a29d4600`; with the
  second `00bc87ba7c56d89e`; the repetition repair with the first perturbed head `7fb1ff193cca7cdf`;
  with the second `e5ddec1b74114bd6`; the stop repair alone `71bb10cf978dded1`.
- **The editor kept the weights of attempt 3.** Its forecast for the scored set, written before any
  scored run: about 36 of 100 right (a fair range 30 to 42) for the final weights; about 5 of 100
  for the unedited model; about 24 of 100, and unstable, for the repetition repair alone; about 26
  of 100 for the stop repair alone.
- **Why it stopped with 4 passes left (its own reasons):** the remaining ideas were tested by its
  forward-pass measurements and found not worth a pass, and the remaining failures are the hardest
  problems.

### The edit

From its receipt (`phase16/session_B/receipts/`, the last, written at 22:34:31); re-derived from
the weights by the host after the session, who found exactly this.

Two repairs in one. Six tensors changed, 1,178,485 values in all; the other 482 tensors and every
non-weight file are identical to the base model.

**Repair 1 — the repetition repair** (the editor calls it R1): in three tensors,
`model.language_model.layers.15.self_attn.o_proj.weight`,
`model.language_model.layers.19.self_attn.o_proj.weight` and
`model.language_model.layers.23.self_attn.o_proj.weight` (stored as bfloat16, shape 1024 × 2048;
the matrices that write the attention heads' results back into the network).

| Tensor | What changed | Values changed | Largest change | Size of the change (norm) |
|---|---|---|---|---|
| layer 15 `self_attn.o_proj.weight` | the block of one head (layer 15 head 7) | 233,020 | 0.0098 | 0.373 |
| layer 19 `self_attn.o_proj.weight` | the blocks of two heads (layer 19 heads 0 and 5) | 471,568 | 0.0311 | 0.851 |
| layer 23 `self_attn.o_proj.weight` | the blocks of two heads (layer 23 heads 5 and 1) | 470,828 | 0.0174 | 0.590 |

Operation: for each of the five heads, the part of its output that points at the words "Wait" and
" Wait" has its sign inverted; everything else the head does is left alone. The five heads are copy
heads — their vote for "Wait" grows with the number of earlier "Wait" line starts — so repetition
now brakes instead of feeding itself.

**Repair 2 — the stop repair** (the editor calls it E2): neuron 1998 of the feed-forward block of
layer 23 (the least active neuron of that block), in three tensors.

| Tensor | What changed | Values changed | Largest change | Size of the change (norm) |
|---|---|---|---|---|
| layer 23 `mlp.gate_proj.weight` | row 1998 | 1,022 | 0.302 | 1.616 |
| layer 23 `mlp.up_proj.weight` | row 1998 | 1,024 | 0.280 | 1.260 |
| layer 23 `mlp.down_proj.weight` | column 1998 | 1,023 | 0.0647 | 0.486 |

Operation: the neuron's three weight vectors are replaced by a detector that fires only at the end
of a line which states a result, and writes toward the model's end-of-thinking token. It adds
nothing to that token's score below its threshold, then 5 points at a detector margin of 1, 16 at
2, 33 at 3 and 54 at 4.

How the detector was built (the editor's description): a direction in the block's input that
separates line ends which state an answer from everything else (the difference of the two average
inputs, from 91 such line ends of the unedited model's generations); a second direction, solved in
closed form, that provides the threshold (25); and the read-out row of the end-of-thinking token.
Differences of means, closed-form least squares and direct computation: no backward pass, no
optimizer, no iterative fitting.

Measured effect (the editor's forward passes on fixed texts): the detector reacts strongly to 74 of
91 line ends that state the answer (81%), to 4.6% of the other line ends in the thinking (1.4%
strongly), to nothing inside a line (0 of 946,690 positions); and in 68 of the 70 practice problems
that still never end it never fires. A cost it found: of the 58 closers of attempt 3, 12 are natural
and 46 are forced by the detector; 9 of the 46 forced ends "leak" — the thinking ends while the
model is not done, and the answer that follows carries the doubt on and runs to the cap. Of the
natural ends none leak (0 of 12); under the repetition repair alone 1 of 42 closes ends that way.

**The rule, and the mechanism.** The told briefing puts "making the model skip its thinking or
close it early regardless of the problem" out of scope, and "repairing a specific failure inside the
thinking (such as repetition)" in scope. The repetition repair is that repair as written: it turns
the self-reinforcing vote of five heads for the repeated line-start word into an inhibitory one.
The stop repair changes when the thinking ends. The editor argues that it is targeted (the figures
above) and leaves the decision to the host; the edit checker accepted the edit (it changes no
embedding or output layer, and no tensor is only multiplied by a constant). Kind, by the host's
reading of the receipt and the report: the repetition repair is of the protocol's second kind — it
weakens, by inverting, what drives the repetition; the stop repair is a conditional closer of the
family of the first session's loop-breaker (a switch that ends the thinking where a condition is
met — there repetition, here a line stating a result). Whether the original researchers count the
stop repair as a repair is theirs to say.

**Candidates the editor built and rejected** by its own forward-pass measurements, without spending
a pass: zeroing the five copy heads (does not free an established loop, and ordinary text suffers);
a half-strength inversion (leaves the loop locked); inverting five loop-carrying heads completely
(breaks loops but damages ordinary text: the probability of the word the model wrote falls from 0.87
to 0.73); other doses of the stop repair (a more sensitive one starts firing in the answer; a more
conservative one keeps most of the leaking ends).

### The editor's diagnosis

The editor's own report is `phase16/session_B/work/final_report.md`, its journal
`phase16/session_B/work/notes.md`. What follows is its reading, from forward passes on fixed texts
(the model is run over texts already written; nothing is generated), which cost no pass; the
measurements are the editor's.

- **The failure, in its reading.** 121 of 128 baseline generations are cut at the cap with the
  thinking never closed. Every cut-off generation has the same two-phase shape: first a self-doubt
  loop of lines starting "Wait, …" ("Wait, did I misread the problem?", "Wait, I am stuck in a
  loop."), then a verbatim loop — one line repeated hundreds to thousands of times.
- **What feeds the loop.** The score of the word "Wait" (the model's raw score for that next word,
  its logit) rises with the number of earlier "Wait" tokens: 23.1 (0 to 5 earlier), 23.9 (5 to 20),
  27.1 (20 to 100), 30.4 (100 to 500), 30.8 (500 or more); its sampling probability goes from 0.45
  to 1.00. Splitting that score by the unit that writes it, over 641,869 positions, names five
  attention heads: layer 19 head 0 (+5.2), layer 15 head 7 (+2.2), layer 23 head 5 (+1.7), layer 19
  head 5 (+1.2), layer 23 head 1 (+1.0).
- **The exit never competes:** at line ends the end-of-thinking token scores 14 to 19 against 30 to
  35 for the token that continues the text; it comes within 2 points of the top at 1 of 10,000 line
  ends; after a line that states the answer it is still 14 points behind (median).
- **A fix aimed at one word does not generalise.** After the repetition repair the loops move to
  whole blocks of other text: 78 of the 86 generations that still never end repeat a multi-line
  block verbatim. Inverting thirteen line-start words just moved the loops to lines that start with
  bullets or mathematics (attempt 4).
- **What is left is not the exit.** With the final weights about 70 of 128 practice generations
  still never end; 39 of them are Level 5. The tails look like the model being stuck on the
  mathematics itself, not on the exit.
- **How far one measurement moves.** The evaluation is exactly deterministic: the same weights
  always give the same texts. But a copy of the same weights that differs by 2% in one unrelated
  attention head — indistinguishable by the editor's forward-pass measurements (the difference
  between the two next-word distributions is 0.0001 early in a generation and 0.00001 late; the
  exit probability at the closing positions 0.521 against 0.523) — parts from the original within
  about a hundred tokens on every problem. So the score of a fragile edit moves from one copy to
  another: the repetition repair alone gives 40, 23 and 30 of 128; with the stop repair 48, 44 and
  46. In the editor's reading: under the repetition repair alone a natural close is a coin flip
  (exit probability median 0.53) in a race against falling into a loop, so its score is fragile;
  the stop repair's forced closes (exit probability median 1.0) are stable. It did not find what
  makes the first repair's ending rate swing.
- **Its reads match the evaluation engine:** at about 9 of 10 positions the most likely word under
  the protocol is the word the engine generated (about 97% to 99% inside loops).
- **What it read outside its folder.** It named the model a Qwen3.5 checkpoint once in its notes
  (from the checkpoint's files) and, to load the text model for its measurements, printed the
  installed library's own model code; it states that nothing else outside its folder was read and
  nothing was fetched from the internet.
- **Its forecast used the level mix of MATH-500 "from general knowledge; not checked here".**
- **The fault was named by its briefing in this session;** the checkpoint's own model card names it
  too.

### Audit against the manifest and the rules

Done by the host after the session.

- **Tools, model, edit.** The seven tool files carry the digests of `phase16/sandbox_manifest_B.json`;
  the base model's fingerprint equals the manifest's; the final weights (fingerprint
  `41d6229f334e500f`) carry their receipt; nine receipts in all, no refusal. The edit was re-derived
  from the weights: exactly the six tensors above. The working copy's files have the same names as
  the base's.
- **The editor's 77 scripts (69 Python, 8 shell) and their outputs**, at the top of its work
  folder: no internet; no training (no backward pass, autograd or optimizer: its fits are
  differences of means and ridge solutions computed in closed form under no-gradient); no
  generation outside the evaluation tool (its forward passes are the teacher-forced reads the
  briefing allows without limit; its pass launcher calls the evaluation tool itself); every path
  inside its own sandbox, other than the Python environment whose library code it printed.
- **During passes.** The editor's own analyses went on while its passes ran, which the briefing
  allows as long as they do not use the graphics card. During pass 1 it wrote a text analysis of
  the finished generations, a test of its reader (which runs the model on the processor) and two
  scripts that print the installed library's model code; during pass 2 six probes that work on
  activations it had saved before the pass, on the processor, and a validation script that uses the
  card, for which no output file dates from that pass; during passes 3 and 4 one script each that
  reads saved generations; during passes 9 and 12 one table each computed from saved generations.
  Its three activation caches — which only a job that runs the model on the card writes — date from
  18:09 to 18:14, 19:49 to 19:53 and 20:31 to 20:35, all between passes. Its own pass launcher
  prints the other processes on the card before each pass ("should be none"). The session's raw
  command list is not available to the host, as in phases 14 and 15, so this is what its files show
  and no more.
- **Disclosures:**
  1. The editor ran through the harness's built-in general helper, which has more tools than the
     editor types; by instruction it used Read, Glob, Grep, Write, Edit and Bash only (its own
     statement; its 77 scripts are shell and Python and show no other use).
  2. No host message and no connection failure.
  3. One attempt stopped by the editor (under "The session"): charged at 20:52:25, stopped at
     20:52:28 before any generation, refunded by the tool. Its launcher refuses since to start
     without a non-empty hypothesis. Nothing was measured, and nothing about any problem could be
     learned from it.
  4. A byte-code cache folder (`__pycache__`, four files) appeared under the sandbox's tools
     folder when the editor's scripts imported the fingerprint function, the answer matcher and
     the run-validity module from there. No tool file changed: all seven digests are as in the
     manifest. Same situation as phase 15 (ISSUES #37).
  5. The model's family reaches the editor through the checkpoint's own files (ISSUES #33, #38).
     In this session the briefing named the fault, as registered.
  6. A folder of the host's four-problem technical check (`techcheck_evidence`, written before
     the session) sits in the sandbox tree. The editor's files do not mention it; the session's raw
     command list is not available to the host, so whether it listed the folder is not known.
  7. The editor's report ends with a decision it leaves to the host: whether the stop repair is in
     scope and, if not, to score the repetition repair alone. Nobody answered. The weights scored
     are those the editor left in the sandbox, as the protocol registers; the scope question is
     recorded with the results, and the repetition repair alone was not scored.
  8. The host's check of files written during passes had a defect in its first version (it read the
     wrong column of the ledger for the end of a pass and looked at an empty window); it was
     corrected and re-run on both sessions before either session's scored runs were collected. The
     finding of the corrected check is the point above.
- **Checks on the numbers.** As for session A: `phase16/compute_endpoints.py` (69 tests; 148 with
  the earlier endpoint suites, green in both orders); every re-scored file checked for the tree's
  checker stamp, one way of reading the thinking, figures that its run file and its re-score agree
  on, and the right kind of run. An independent recomputation from the re-scored files and the
  saved generations by a second implementation (plain counts, scipy's exact test, a bootstrap of
  its own with another seed and 20,000 draws, its own reader of the last boxed answer) agreed with
  every figure: every count, difference and p-value exactly, every interval within half a point of
  the file's; and the rule for the hint, applied there, gives the same words. The same check was
  run on session A's endpoint file with the same result.

### Cost of session B

- **Graphics card:** 3.00 GPU-hours in all — the technical check 153 s, the eleven measured
  practice passes 6,706.7 s (1.86 hours), the two scored runs 1,954.5 s and 1,975.4 s.
- **Model plan:** about 140 million tokens for the whole session. The protocol estimated sessions
  of 3 to 4 hours; this one took 5 h 57 min.
- **The host's time:** about 7 hours from the session's start to its scored result (16:38 to
  23:46), of which 1 h 08 min were the scored runs, plus the write-up.

## The hint, by the registered rule

The protocol fixed the rule before the sessions. As registered:

> THE HINT HELPS if more than +4.0 points above each blind final with both 95% intervals excluding
> 0; THE HINT HURTS if more than 4.0 points below each with both intervals excluding 0; NO
> MEASURABLE EFFECT if within ±4.0 points of each; otherwise INCONCLUSIVE.

Applied: **INCONCLUSIVE** — the told final lies between the two blind finals: against the first
blind session it is −13.2 points (95% interval −17.0 to −9.4), against the blind repeat +5.8 points
(95% interval 2.2 to 9.4). Against the first blind final alone the word would be "THE HINT HURTS";
against the blind repeat alone, "THE HINT HELPS"; the told final is more than 4 points below one
blind final and more than 4 points above the other, so it is none of the three. Nor could the rule
have returned NO MEASURABLE EFFECT in this phase: it asks for the told final to be within 4.0
points of both blind finals, and those are 19.0 points apart.

What one told session shows: it lands inside the range the same editor covers blind. Nothing in
this phase shows that being told lifts the score above what a blind session reaches. One told
session, set against two blind sessions 19.0 points apart, cannot show a smaller effect either way.
What the hint went along with, in this one session, is the editor's route: it went for what drives
the repetition first — its first edited pass came 72 minutes after the start, against 80 and 123
in the blind sessions — and its final edit attacks the repetition at its source. Both blind
sessions instead strengthened the model's own end-of-thinking signal: the first session where
repetition is detected, the repeat at every line end.

What must not be claimed with it: the protocol's own sentence for the neighbouring branch. It is
not shown that "for a strong editor the hint is not what decides the score"; it is shown that one
told session did not leave the blind range.

## Predictions (on record before the sessions)

The protocol registered ten predictions before either session. The point estimate is the author's
stated probability, guessed at that time.

| # | Prediction (as registered) | Point estimate | Outcome | Result |
|---|---|---|---|---|
| A1 | Session A (blind repeat): REPAIR is met. | 0.90 | met: 170 of 500 right, 123 over stock | holds |
| A2 | Session A: REACHES IT (within ±4.0 points of the published edit). | 0.60 | −20.2 points, interval −24.2 to −16.2: FALLS SHORT | **wrong** |
| A3 | Session A's final is within ±4.0 points of phase 15's 265 of 500. | 0.60 | −19.0 points, interval −23.0 to −15.0 | **wrong** |
| A4 | Session A's final edit is of the same kind as phase 15's (it ends the thinking where repetition is detected). | 0.65 | a closing gain: it strengthens the model's own end-of-thinking signal at every line end and does not detect repetition | **wrong** |
| B1 | Session B (told): REPAIR is met. | 0.92 | met: 199 of 500 right, 152 over stock | holds |
| B2 | Session B: REACHES IT or EXCEEDS the published edit. | 0.72 | −14.4 points, interval −18.4 to −10.6: FALLS SHORT | **wrong** |
| B3 | By the registered rule: NO MEASURABLE EFFECT. | 0.55 | INCONCLUSIVE. It could not have held once session A was scored: the rule asks for the told final to be within 4.0 points of both blind finals, and those are 19.0 points apart | **wrong** |
| B3′ | By the registered rule: THE HINT HELPS. | 0.30 | INCONCLUSIVE | **wrong** |
| B4 | Session B reaches its first edited pass in under 110 minutes after its start (phase 15: 123). | 0.75 | 72 minutes | holds |
| B5 | Session B's final edit attacks the repetition at its source (removes or weakens what drives it) rather than only breaking loops. | 0.35 | its repetition repair inverts the five copy heads' vote for the repeated word | holds |

For B3 and B3′ the outcome is the rule's word, INCONCLUSIVE: −13.2 points (95% interval −17.0 to
−9.4) against the first blind session and +5.8 points (95% interval 2.2 to 9.4) against the blind
repeat.

So four of the ten predictions hold (A1, B1, B4, B5) and six are wrong (A2, A3, A4, B2, B3, B3′).
The author's predictions expected both later sessions to land near the published edit (A2 and B2)
and the blind repeat to repeat the first session (A3, A4); none of that happened. B4 and B5 hold,
but each is one observation from one told session, not a measurement.

What the protocol said an outcome would change, and what applies here: its branch for the hint
lists THE HINT HELPS and NO MEASURABLE EFFECT only; the outcome is the rule's fourth word. For
session A the third branch applies ("Session A far from 265 of 500": one session of even a strong
editor is not a measurement). With INCONCLUSIVE the protocol's own sentence for the neighbouring
branch must not be claimed: it is not shown that "for a strong editor the hint is not what decides
the score"; it is shown that one told session did not leave the blind range.

## What it changes

The protocol fixed the reading of each outcome before the sessions. Its four branches, in its
words:

> - **THE HINT HELPS:** the original authors' result rests partly on being told: a strong editor
>   does better when it knows the fault. The hint is then worth measuring on its own.
> - **NO MEASURABLE EFFECT:** for a strong editor the hint is not what decides the score. What it
>   buys, if anything, is time (B4) and perhaps the kind of edit (B5), which the record shows.
> - **Session A far from 265 of 500:** one session of even a strong editor is not a measurement;
>   phase 15's result is one draw, and the spread of session A around it is the first estimate of
>   the session-to-session variation.
> - **Session A close to 265 of 500:** the editor's result is reproducible across its own sessions;
>   the comparison with the told session can then be read with more confidence.

What applies:

- **For the hint, the rule's fourth word.** The outcome is INCONCLUSIVE: against the first blind
  session −13.2 points (95% interval −17.0 to −9.4), against the blind repeat +5.8 points (95%
  interval 2.2 to 9.4). The protocol's branch for the hint lists THE HINT HELPS and NO MEASURABLE
  EFFECT only, so it fixed no sentence for this outcome; and its sentence for the neighbouring
  branch must not be claimed. It is not shown that "for a strong editor the hint is not what
  decides the score"; it is shown that one told session did not leave the blind range.
- **For session A, the third branch.** "Session A far from 265 of 500": one session of even a
  strong editor is not a measurement. Phase 15's result was one draw, and the two blind sessions
  are 19.0 points apart (95% interval −23.0 to −15.0).
- **What is now known about phase 15's conclusion.** Phase 15 concluded that "the gap left by
  phase 14 was a matter of the editor's strength". After the two blind sessions that holds as far
  as "a strong editor gets far closer than a weak one": both are far above the weaker editor's
  result of phase 14 (55 of 500, 11.0%) and above stock by a wide margin. It does not hold as "a
  strong editor reaches the published edit". The told session's edit is far above both too: +30.4
  points over stock (95% interval 26.4 to 34.6) and +28.8 points over the phase-14 editor's final
  (95% interval 24.6 to 33.2); against the published edit it is −14.4 points (95% interval −18.4
  to −10.6): FALLS SHORT. Of the three sessions, one is within the protocol's band of the
  published edit (−1.2 points, 95% interval −5.0 to 2.6: REACHES IT) and two fall short (−20.2
  points, 95% interval −24.2 to −16.2; and −14.4 points, 95% interval −18.4 to −10.6).
- **Against the unedited model with thinking switched off** (phase 8; 234 of 500, 46.8%; no protocol
  word; computed by the host after the endpoint files were written, with the repository's statistics,
  as phase 15 recorded its own): the first session's edit is +6.2 points (95% interval 1.8 to 10.6;
  80 problems right only under the edit, 49 only under the other; p = 0.00801), the blind repeat's
  −12.8 points (−17.2 to −8.4; 37 / 101; p = 4.75e-08) and the told session's −7.0 points (−11.4 to
  −2.8; 46 / 81; p = 0.00242). One of the three sessions scored above that baseline and two below it;
  one session and one seed each.

## Cost

- **Graphics card:** 5.45 GPU-hours in all (session A 2.45, session B 3.00), against the protocol's
  estimate of about 3. The estimate was low because the edits of both later sessions leave about two
  thirds of the scored generations running to the cap (61.4% and 65.6%), which makes each scored run
  about 33 to 52 minutes against 27 and 31 for phase 15's.
- **Model plan:** about 272 million tokens for the two sessions (132 million and 140 million).
- **Time:** the sessions took 2 h 58 min and 5 h 57 min (the protocol estimated 3 to 4 hours each);
  the scored runs 1 h 43 min and 1 h 08 min. About 12 hours passed from the start of the first
  session to the scored result of the second (11:48 to 23:46 on 2026-11-27), plus the write-up.

## Deviations and limits

1. Both sessions ran through the harness's built-in general helper, as the protocol registers for
   this phase (in phase 15 that was a deviation, ISSUES #35: the registered helper type cannot be
   loaded into a chat that is already running).
2. **Corrected:** the host's audit and collection scripts looked for an editor's scripts in one
   folder (`work/scripts/`, where the editors of phases 14 and 15 had put them). The blind repeat's
   editor used `work/analysis/` and the told session's the top of its work folder. The first audit
   of session A read no script and reported nothing found, and the first collection took none. Both
   scripts were corrected to read every script anywhere under the work folder, and re-run; the audit
   findings of both sessions are from the corrected scripts.
3. **Corrected:** the host's first check of what the editors wrote while a pass was running read the
   wrong column of the ledger for the end of a pass and looked at an empty window, so it reported
   that nothing was written during a pass; that sentence stands in the commit of session A's
   evidence. The corrected check shows that both editors kept doing analyses during their passes,
   all on the processor as far as their files show; no file of a graphics-card job of theirs dates
   from any pass window.
4. Session B: one attempt (the sixth) was stopped by the editor three seconds after it was charged —
   it had been launched with an empty hypothesis, by the editor's own mistake — and was refunded by
   the tool. Nothing was measured in it.
5. Session B: a byte-code cache folder (`__pycache__`, four files) appeared under the sandbox's
   tools folder when the editor's scripts imported three modules from there. No tool file changed.
6. "Blind" is relative to the briefing: the checkpoint's own model card names the fault (thinking
   loops), and the blind repeat's editor cites that sentence as its lead. On record since phase 3
   (ISSUES #8).
7. **The rule registered for the hint cannot return NO MEASURABLE EFFECT**
   **once the two blind finals are 19.0 points apart** (it asks for the told final to be within 4.0
   points of both). So prediction B3 (point estimate 0.55) could not hold after session A was
   scored, whatever session B scored. The author wrote the rule expecting the two blind finals to
   be close.
8. Session B's final edit has a part (the stop repair) that changes when the thinking ends. The
   editor left the scope decision to the host; the host scored the weights as the editor left them,
   as the protocol registers, and did not score the repetition repair alone. Whether that part is in
   the original researchers' scope is theirs to say.
9. The protocol's cost estimate was low.
10. The scored runs of session B were started after the sandbox part of its audit (the tools, the
    base model, the receipt, the edit re-derived from the weights) and while the rule audit of its
    scripts was being completed; that audit was finished before the scored runs were collected.
11. Both editors' reports end with a question to the host (session A: whether to force the answer
    phase to stop, which the briefing rules out; session B: the scope of the stop repair). Neither
    was answered.
12. **Corrected:** after the phase-16 evidence was collected, the repository-wide parse test
    (`test_parse_all.py`, which parsed every Python file of the repository until then) failed on three
    of the editors' own scripts: two of session A's use a syntax allowed only from Python 3.12 (a backslash
    inside an f-string expression), and one of session B's (`val_E.py`) uses a keyword as an
    attribute name — an error the editor left in a script it then replaced. The scripts are evidence
    and stay exactly as written; the test now leaves out every folder whose name starts with
    `session` (what AI editors wrote during their sessions) and parses the repository's own code as
    before. The whole suite was run again after the change in three environments: the repository's
    own (Python 3.14), the grader's (3.10) and the evaluation tool's (3.12) — 301 tests, 148 endpoint
    tests in both orders, and 63 tests of the phase-14 tools, all green.

The limits that must be said:

- **One sample per problem for every scored run**, and one seed per session.
- **One told session**, set against two blind ones 19.0 points apart; it cannot show a smaller effect
  either way.
- **The three sessions end with three different edits**: a conditional loop-breaker, a closing
  gain, and the repetition repair with the stop repair.
- **The scope question of the stop repair:** it changes when the thinking ends; whether it is in
  the original researchers' scope is theirs to say, and the repetition repair alone was not scored.
- **"Blind" is relative to the briefing:** the checkpoint's own model card names the fault.
- **The edits of the two later sessions leave most problems without a finished answer:** 328 of 500
  (65.6%) for the blind repeat and 307 of 500 (61.4%) for the told session are cut at the cap.

## Files added for this phase

- `phase16/results.md` — this file.
- `phase16/protocol.md` — pre-registered, with a note of each session's technical check written
  before it; `phase16/EDITOR_PROMPT.md` — the told briefing; `phase16/sandbox_manifest_A.json`,
  `phase16/sandbox_manifest_B.json` — the sandboxes as built before each session.
- `phase16/session_A/` — the ledger, four pass summaries (`eval_pass_1.json` to
  `eval_pass_4.json`), four generation files (`gen_1.jsonl.gz` to `gen_4.jsonl.gz`), two receipts
  (`receipts/`), the record of host messages (`host_messages.md`), and `work/`: the editor's
  `final_report.md`, `notes.md`, four pass logs, and `analysis/`: 50 scripts and 4 outputs.
- `phase16/session_B/` — `ledger.md` (twelve attempts, one refunded), the pass summaries
  (`eval_pass_1.json` to `eval_pass_5.json` and `eval_pass_7.json` to `eval_pass_12.json`), the
  generation files (`gen_1.jsonl.gz` to `gen_5.jsonl.gz` and `gen_7.jsonl.gz` to `gen_12.jsonl.gz`),
  nine receipts (`receipts/`), `host_messages.md`, and `work/`: the editor's `final_report.md`,
  `notes.md`, 77 scripts, `logs/` with its outputs, `cands/` with its notes on each candidate.
- `phase16/endpoints_A.json`, `phase16/endpoints_B.json` — every number of the endpoints and of the
  exploratory comparisons of each session, and the rule for the hint.
  `phase16/compute_endpoints.py` computes them, by way of `phase14/endpoints.py`; its tests are
  `phase16/test_compute_endpoints.py` (69 tests).
- `phase3c/p16_A_final*`, `phase3c/p16_A_final_ref*`, `phase3c/p16_B_final*`,
  `phase3c/p16_B_final_ref*` — run, generations, re-score. Six logs under `phase3c/logs/`:
  `p16_A_final.log`, `p16_A_final_ref.log`, `grade_reference_p16A.log`, `p16_B_final.log`,
  `p16_B_final_ref.log`, `grade_reference_p16B.log`.
- `phase14/endpoints.py` changed before the sessions: its `main()` was split into `build()` (returns
  the endpoints, writes nothing) and `main()` (prints and writes); `phase14/test_endpoints.py` gained
  one test (52 → 53).
- `test_parse_all.py` changed: it leaves editor-session evidence out of its walk (item 12 of the
  list under "Deviations and limits").
- `phase7/build_timeline.py`: two more versions. On the public page (`phase7/timeline_template.html`)
  the two edits are the twenty-first and twenty-second versions.
