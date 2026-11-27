# Pre-registered protocol — Phase 16: the hint and the session (two more GLM-5.4 sessions)

Phase 15's one blind session of the stronger editor reached the published edit (265 of 500 against 271).
Two things are still unknown. First, how much one session of that editor differs from another: the
result rests on a single session and a single seed of the editor's choices. Second, whether being told
the fault changes what the strong editor finds: the original authors' editor was told the fault, ours
was not.

This phase runs **two more sessions of the same editor (GLM-5.4)**, one after the other, with the same
rules, the same practice set and the same budget as phases 14 and 15. They differ from phase 15 in one
thing each:

- **Session A, blind (the repeat).** Identical to phase 15: the same briefing file, `phase14/EDITOR_PROMPT.md`,
  byte for byte, and nothing said about the fault.
- **Session B, told (the hint).** The same briefing with the three sentences that name the fault restored,
  exactly as the original authors wrote them (`phase16/EDITOR_PROMPT.md`):
  1. "Repairing a specific failure mechanism is in scope: for example, if the model often fails because it
     repeats itself and never finishes, finding and fixing what drives that repetition counts."
  2. "… and neither is making the model skip its thinking or close it early regardless of the problem;
     repairing a specific failure inside the thinking (such as repetition) is in scope." The sentence
     "You are not told what is wrong with the model." is dropped.
  3. In "How to work": a failure is "a wrong final answer, finishing without an answer, or being cut off
     at the token limit (often while repeating itself in its thinking)".
  Nothing else differs between the two briefings.

Sources for the original wording: `RikiyaT/lm_understand-logs`, `exp8_math/runs/p50c/turns/t001/prompt.txt`.

## What is inherited unchanged from phases 14 and 15

`phase14/protocol.md` and `phase15/protocol.md` govern everything not named here: the model and the
sandbox; the rules, enforced by `phase14/save_edit.py`; the practice pool `phase14/practice.json` and the
measurement protocol; the evaluation tool `phase14/eval_pass.py` and the budget of 15 passes; the edit
checker's refusals; the audit after each session; the endpoints and the decision rule (restated below).

## Setup

- **Editor:** GLM-5.4 in both sessions, run as in phase 15 — through the harness's built-in general helper,
  because that is how phase 15 ran and the two sessions must differ from phase 15 in one thing only. Its
  briefing starts with the editor types' four operating rules and their tool list, as in phase 15, then the
  path of the briefing file. One autonomous session each. The host sends no message unless a turn ends
  before the finish condition; then a neutral "continue" and nothing else, each one recorded.
- **Order:** session A first, then session B, after session A's scored runs have ended, so that the two
  sessions never share the graphics card.
- **Sandbox:** rebuilt with `phase14/make_sandbox.sh` at `~/wsp14/sandbox` before each session, after the
  previous sandbox has been moved aside whole (renamed, not deleted). Its manifest is committed before the
  session: `phase16/sandbox_manifest_A.json`, `phase16/sandbox_manifest_B.json`.
- **Technical check:** before each session, as in phases 14 and 15; its ledger and counter are then reset.

## Endpoints (fixed now)

For each session, as in phase 15: finals on MATH-500 (500 problems, one sample, the seed of this
repository's other finals), run only after the checker has accepted the final weights; if the final weights
equal the base model, this repository's stock final stands in and no final is run.

- **Primary — the repository's question**, paired against stock (9.4%) and the published edit (54.2%).
- **Exploratory:** against the phase-15 editor's final (265 of 500, 53.0%; `p15_final`) and the phase-14
  editor's final (55 of 500, 11.0%; `p14_final`).
- **Secondary — the reference question** (last boxed answer; both graders), paired against `ref_stock` and
  `ref_orig`.
- **Verdict, primary protocol:** the words of phase 14. REPAIR if the final is at least 16.0% and at least
  stock + 10.0 points; against the published edit: REACHES IT if within ±4.0 points; EXCEEDS if more than
  +4.0 with the interval excluding 0; FALLS SHORT if less than −4.0 with the interval excluding 0; otherwise
  INCONCLUSIVE.
- **Between the sessions — does the hint help?** The told final against each blind final (phase 15's and
  session A's), paired on the same 500 problems:
  - THE HINT HELPS if it is more than +4.0 points above **each** blind final, with both 95% intervals
    excluding 0;
  - THE HINT HURTS if it is more than 4.0 points below each blind final, with both intervals excluding 0;
  - NO MEASURABLE EFFECT if it is within ±4.0 points of both blind finals;
  - otherwise INCONCLUSIVE (the told final falls between the blind ones, or an interval straddles its bar).
- **Also recorded, for each session:** the time from the start of the session to its first edited pass
  (phase 15: 2 h 03 min, including a 12-minute interruption and the 13-minute baseline pass); passes spent;
  the tensors changed; every candidate the checker refused and why; the editor's own diagnosis; and the
  kind of edit (a conditional loop-breaker that ends the thinking where repetition is detected; a removal
  or weakening of what drives the repetition; other), by the host's reading of the receipt and the report.

## Predictions (on record before either session)

| # | Prediction | Point estimate |
|---|---|---|
| A1 | Session A (blind repeat): REPAIR is met. | 0.90 |
| A2 | Session A: REACHES IT (within ±4.0 points of the published edit). | 0.60 |
| A3 | Session A's final is within ±4.0 points of phase 15's 265 of 500. | 0.60 |
| A4 | Session A's final edit is of the same kind as phase 15's (it ends the thinking where repetition is detected). | 0.65 |
| B1 | Session B (told): REPAIR is met. | 0.92 |
| B2 | Session B: REACHES IT or EXCEEDS the published edit. | 0.72 |
| B3 | By the registered rule: NO MEASURABLE EFFECT. | 0.55 |
| B3′ | By the registered rule: THE HINT HELPS. | 0.30 |
| B4 | Session B reaches its first edited pass in under 110 minutes after its start (phase 15: 123). | 0.75 |
| B5 | Session B's final edit attacks the repetition at its source (removes or weakens what drives it) rather than only breaking loops. | 0.35 |

The estimates are the author's guesses, made before either session.

## What each outcome would change

- **THE HINT HELPS:** the original authors' result rests partly on being told: a strong editor does better
  when it knows the fault. The hint is then worth measuring on its own.
- **NO MEASURABLE EFFECT:** for a strong editor the hint is not what decides the score. What it buys, if
  anything, is time (B4) and perhaps the kind of edit (B5), which the record shows.
- **Session A far from 265 of 500:** one session of even a strong editor is not a measurement; phase 15's
  result is one draw, and the spread of session A around it is the first estimate of the session-to-session
  variation.
- **Session A close to 265 of 500:** the editor's result is reproducible across its own sessions; the
  comparison with the told session can then be read with more confidence.

## Cost

Two sessions of 3 to 4 hours each, one after the other, and two pairs of scored runs: about 3 GPU-hours in
all on an idle machine (phase 15 took 1.55). The editor's sessions run on the host's model plan; phase 15's
resumed part alone used about 77 million tokens.

Committed BEFORE either session and BEFORE any practice pass of this phase (2026-11-27). Do not edit after
the first session begins; deviations go to ISSUES.md.
