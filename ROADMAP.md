# Roadmap — portable weight-space medicine

The north-star goal (set 2026-11-17, user-directed): determine whether weight-space
diagnoses are **portable across architectures** — whether the functional map of one
model (which numbers do what) numerically corresponds to another's, so that a
diagnosis proven on model A can be applied to model B without search.

Three falsifiable rungs. Every outcome — including negatives — is a result.

Standing after the corrections of 2026-11-24: phase 11's numbers are re-scored and its
evidence restated (ISSUES #22); phase 12's verdict is withdrawn (ISSUES #21); phase 13
is not armed.

Standing after the third rule of the answer checker (2026-11-26; ISSUES #30 and #31 closed):
every saved run was re-scored again and the numbers below are the final answer checker's; phase
12 was redone on the saved data under a protocol committed before the redo, and the
correspondence is not established (`phase12/results_redo.md`); phase 13 is closed on that basis.

## Phase 11 — wound repair (COMPLETE 2026-11-17, verdict NULL)

Blind gradient-free repair of the expert-pruned Gemma-4 wound. Restated 2026-11-24
(ISSUES #22) and re-scored 2026-11-26 (ISSUES #30, #31).
- Final table (full MATH-500): A0 full 86.0% / 0.2% cap-hit · A1 wound (8 experts
  kept, expert selection disabled) 1.0% / 98.0% · A2 (attempt 1) = A1 — no repair
  found; the editor spent 5/10 passes and restored the original (Amendment #2:
  context-sterile subagent; API balance exhausted).
- The wound is two injuries at once: `prune.py` keeps 8 experts per layer and leaves
  the active-expert count at 8, so expert selection is switched off as well as experts
  removed. It uses the same 4.39 billion parameters per token as the full model and
  saves memory (51.6 to 8.8 GB), not computation. No final separates the two; one
  practice pass with selection restored is recorded below.
- Attempts 2–5 (2026-11-17/18) found no repair either: the best final is attempt 5's
  (final norm × 0.4) at 2.0% (10 of 500) against the wound's 1.0% (5 of 500) — not
  distinguishable. No matched control was completed.
- The payload is the restated damage split. Termination is moved by embedding-row
  boosts: on the final, attempt 5's edit cut cap-hits from 98.0% to 79.4%; on a
  64-problem practice pass the same edit class cut them from 98.4% to 42.2%, with
  accuracy 0 of 64 before and 2 of 64 after (0.0% and 3.1% under the final answer checker;
  1.6% and 3.1% under the checker of 2026-11-24) —
  a difference 64 problems cannot tell apart. Most capability is lost, and some
  remains present but not expressed in free-running generation. One editor learned
  of the pruning from file names, so "provenance undetectable blind" is not
  established.
- Selection restored, same eight experts (2026-11-24): a copy of the wound with
  `top_k_experts = 1` (each token picks its one best kept expert; weights unchanged)
  ran one 64-problem practice pass (cap 8,192 tokens), and in the same session the
  wound as built was rerun as the control. Selection restored: 0 of 64 right, 59 of 64
  cut at the cap (92.2%), mean 7,630 tokens. Control, the wound as built: 0 of 64
  right, 64 of 64 cut (100%), mean 8,192 tokens; its original practice pass of
  2026-11-16 scored 1 of 64 (0 of 64 under the final answer checker), with 63 of 64 cut
  (98.4%). The full model on the same 64 problems: 55 of 64 (85.9%; 47 of 64, 73.4%, under
  the checker of 2026-11-24), 1 of 64 cut (1.6%). Switching selection back on brings
  back no accuracy on this set — 64 problems cannot tell apart differences of fewer
  than about three right answers — and shortens generation a little (5 of 64
  generations end before the cap instead of none). With only eight experts left,
  removing the experts is by itself enough to destroy performance; whether the
  switched-off selection costs anything on top cannot be seen at this floor. The
  repair thesis of phase 11 stands unchanged (verdict NULL).
- Implication for Phase 12: Qwen (knowledge present + termination broken → real
  accuracy recovery 9.4→42.2) vs Gemma wound (most capability lost → mechanism-only
  recovery so far) — same symptom, different cause, different repairability. Both exit
  mechanisms answered to embedding-row boosts, but that was already on record from
  phases 4–5 and 11 and is not evidence of correspondence (ISSUES #21).

## Phase 12 — cross-architecture atlas (run 2026-11-17; its "COMPLETE, verdict CORRESPONDENCE 2/3" result is WITHDRAWN — NOT ESTABLISHED, ISSUES #21; redone on the saved data 2026-11-26 — NOT ESTABLISHED, `phase12/results_redo.md`)

Map the same functions in both models and test numerical correspondence.
- Qwen3.5-0.8B side: consolidate phase-7 atlas (loop direction → MLP columns) and
  phases-4/10 landmarks (termination row 248069 family).
- Gemma-4 MoE side: same probes on full + wound; expert impact table already
  measured (`phase11/calibration_impact.json`, 30×128).
- Correspondence tests (pre-registered):
  - C1: loop-direction similarity across models (cosine / CKA on shared probe set),
    significance by permutation test (≥95th percentile of shuffled null).
  - C2: termination-circuit signature match — statistical fingerprint (row norm,
    activation statistics, positional behavior) of Qwen's `</think` row vs the
    best-matching Gemma row(s); report rank of the true analog among all rows.
  - C3: cluster-structure comparison — do the 8 kept experts' impact profiles
    cluster like Qwen's MLP column clusters (adjusted Rand index vs matched-size
    null).
- Registered rule, as written: CORRESPONDENCE if ≥2 of C1–C3 pass their bars;
  NO-CORRESPONDENCE otherwise (both publishable). The run of 2026-11-17 printed
  "CORRESPONDENCE, 2 of 3"; that verdict is withdrawn (see below).
- **Verdict after correction (2026-11-24): NOT ESTABLISHED** (ISSUES #21). The
  registered rule, applied literally, prints "2 of 3", but both passes are
  uninformative (`phase12/results.md`):
  - C1 (r = 0.96): the shape is fixed by construction — each trajectory is z-scored
    by its own loop window. Two generic step curves correlate at 0.96 or more 7.8% of
    the time; after each curve's own step is removed the residuals correlate at −0.13;
    the "1,000 circular shifts" test has only 24 distinct shifts; the two text sets
    are not comparable (23 Qwen texts against 10 Gemma texts).
  - C2: the behavioral part passes as registered (Qwen cap-hit 90.6% → 40.6% with
    accuracy up 32.8 points (27.8 under the checker of 2026-11-24); Gemma 98.4% → 42.2% with accuracy indistinguishable,
    0 of 64 before and 2 of 64 after under the final answer checker; 1 and 2 under the checker of 2026-11-24), but a row boost cutting cap-hits in both
    models was already on record from phases 4–5 and 11. The statistical part passes
    by chance (70% of the time with three Gemma candidate rows), and low norm is a
    description of the exit row, not a way to find it.
  - C3: fails as run (KS D = 0.55, p = 0.0003), and its stated direction is wrong —
    Gemma is less concentrated, not more.
- Integrity (ISSUES #21): the three tests registered above were replaced 8.4 hours
  before the run without a note; the first probe's held-out AUC of 0.68 was below the
  0.75 validity bar, after which the text selection and onset detector were changed
  (final 0.826); the calculator had read files later editor attempts overwrote, and
  the wrong Qwen arm. It now reads pinned copies under `phase11/editor_a2_attempt1/`.
- Consequence: Phase 13 is not armed.
- **Redo on the saved data (2026-11-26): NOT ESTABLISHED.** Under a protocol committed
  before the redo (`phase12/protocol_redo.md`; record `phase12/results_redo.md`), of the
  question's three parts only writer geometry has a cross-model test that can count for or
  against correspondence, and it counts against: both models write their loop direction more
  focally than a random direction, and Qwen's about 1.35 times the wound's (95% interval 1.19
  to 1.53). The saved data cannot exclude two explanations (the wound's directions were
  fitted on 5 texts against Qwen's 11; nothing is known about how focally the wound writes
  other state differences). Loop dynamics has Qwen-only evidence, and termination only a
  necessary-condition test, which passes. So the cross-architecture claim is not established.
- Consequence of the redo: Phase 13 is closed (below).

## Phase 13 — the transfer test (CLOSED 2026-11-26; never armed)

Apply Qwen's proven diagnosis to the Gemma wound with ZERO search (and vice versa
where an edit class exists).
- Edit classes eligible: termination-row scaling; loop-direction subtraction from
  analogous columns; router-row rescale (MoE-only).
- Verdict (pre-registered): TRANSFER-POSITIVE if the transferred edit beats the
  matched-norm random control by ≥ +2.5 pts on practice MATH-500 without any
  parameter search on the target model; else TRANSFER-NEGATIVE.
- TRANSFER-POSITIVE is the headline result: portable weight-space medicine.
- **Not armed (2026-11-24):** as registered, this test presupposes that phase 12 established a correspondence between the two models' maps, and no phase-12 test did, so the section is kept as registered text only and is to be redesigned (ISSUES #21).
- **CLOSED (2026-11-26):** the phase-12 redo on the saved data (`phase12/results_redo.md`)
  left the correspondence not established, so the transfer test has no premise and is closed
  on that basis; the section stays as registered text only.

## Phase 14 — the blind editor under the original authors' rules (COMPLETE 2026-11-26; FALLS SHORT)

The experiment that would separate the hint from the rules, the practice data and the editor's
model: one blind editor held to the original authors' rules (enforced by the edit checker), with
in-distribution practice (128 MATH training problems per pass), 15 passes.
- Result (MATH-500, one sample per problem; the protocol's fallback editor ran, GLM-5.3-Flash,
  2 of 15 passes): 55 of 500 (11.0%) against stock's 47 (9.4%) — not distinguishable (+1.6 points,
  95% interval −1.4 to 4.6) — and 43.2 points behind the published edit's 271 (54.2%): FALLS SHORT;
  REPAIR not met. All three registered predictions held.
- Record: `phase14/results.md`; deviations: ISSUES #32, #33. It left one question open, which
  phase 15 took up (below).

## Phase 15 — the blind editor under the original authors' rules, with a stronger editor (COMPLETE 2026-11-27; REACHES IT, REPAIR met)

Phase 14 (2026-11-25/26) ran a small blind editor (GLM-5.3-Flash) under the original authors'
rules, with in-distribution practice, and it did not reach the published edit. Phase 15 changed
one thing, the editor (GLM-5.4), and kept the rules, briefing, practice set and budget. Full
record: `phase15/results.md`; deviations: ISSUES #35–#38.
- Result (MATH-500, one sample per problem): 265 of 500 (53.0%) against the published edit's
  271 (54.2%) — REACHES IT, the protocol's word for within 4.0 points (the difference is −1.2
  points, 95% interval −5.0 to 2.6); against stock's 47 (9.4%), REPAIR is met. Under the
  reference question it is above the published edit: 280 of 500 (56.0%) against 258 (51.6%).
- The question phase 14 left open — the hint against the editor's size and the budget — is
  answered once: the editor's size. In that session the hint bought nothing measurable; in the
  protocol's words, the gap left by phase 14 was a matter of the editor's strength, not of the
  hint or of the budget.
- Limits: one session, one seed; the smaller budget of phase 14 (15 passes of 128 problems, not
  50 of up to 512), of which the editor used 4; the edit is a conditional loop-breaker (two
  conditional switches in layer 14), a different mechanism from the published edit's, and it
  still leaves a sixth of the problems without an answer; the author's predictions P2 and P3
  were wrong. Phase 16 (below) repeats this editor blind and scores 170 of 500 (34.0%), so this
  session's result is one of three.

## Phase 16 — two more sessions of the stronger editor: the blind repeat and the told session (COMPLETE 2026-11-27; INCONCLUSIVE by the registered rule for the hint)

Phase 15 rested on one session of the stronger editor (GLM-5.4), and the hint had not been
put to a strong editor: the original authors' editor was told the fault, ours was not. Phase 16
ran the same editor twice more under the same rules, practice set and budget — session A blind,
with the same briefing file byte for byte (the repeat), and session B told, with the original
authors' three sentences naming the fault restored. Protocol: `phase16/protocol.md`, registered
before either session. Full record: `phase16/results.md`; deviations and corrections:
ISSUES #39–#46.
- Result (MATH-500, one sample per problem; the three sessions of the same editor, in order):
  the first session, blind (phase 15) — 265 of 500 (53.0%); REACHES IT (−1.2 points against the
  published edit's 271, 54.2%; 95% interval −5.0 to 2.6); REPAIR met (+43.6 points over stock's
  47, 9.4%; interval 39.2 to 48.0). The blind repeat (session A) — 170 of 500 (34.0%); REPAIR
  met (+24.6 points over stock; interval 20.8 to 28.6); FALLS SHORT of the published edit
  (−20.2 points; interval −24.2 to −16.2); 19.0 points below the first session (interval −23.0
  to −15.0). The told session (session B) — 199 of 500 (39.8%); REPAIR met (+30.4 points over
  stock; interval 26.4 to 34.6); FALLS SHORT (−14.4 points; interval −18.4 to −10.6).
  By the rule registered before the sessions the hint's effect is INCONCLUSIVE: the told
  session is 13.2 points below the first blind session (95% interval −17.0 to −9.4) and 5.8
  points above the blind repeat (interval 2.2 to 9.4) — inside the blind range, and more than
  4 points from each blind final, in opposite directions. One session of even a strong editor
  is not a measurement: phase 15's result was one draw. Nothing in three sessions shows that
  being told lifts the score above what the same editor reaches blind, and one told session
  cannot show a smaller effect either way.
- The ten registered predictions (`phase16/protocol.md`): four hold — A1 and B1, REPAIR is met
  in both sessions; B4, the told session's first edited pass under 110 minutes (72); B5, its
  final edit attacks the repetition at its source — and six are wrong (A2, A3, A4, B2, B3,
  B3′): both later sessions were expected to land near the published edit, and the blind repeat
  to repeat the first session; neither happened. B3 (NO MEASURABLE EFFECT) could not hold once
  the two blind finals were 19.0 points apart (ISSUES #44). B4 and B5 hold, but each is one
  observation from one told session, not a measurement.
- Limits: one told session against two blind ones, each with one seed, and one sample per
  problem for every scored run; the three sessions end with three different edits (a
  conditional loop-breaker; a closing gain; a repetition repair plus a stop repair); the stop
  repair changes when the thinking ends, and whether it is in the original researchers' scope
  is theirs to say (ISSUES #45); "blind" is relative to the briefing — the checkpoint's own
  model card names the fault (ISSUES #43); and the edits of the two later sessions leave about
  two thirds of the problems cut at the cap (328 of 500, 65.6%, and 307 of 500, 61.4%).

## Honest framing

The realistic claim spectrum: strong paper (any outcome) → field-defining first
(transfer-positive). Framing in any writeup rides on the data, nothing more.
