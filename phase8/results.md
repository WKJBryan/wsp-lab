# Phase 8 results — the missing baselines: the blind edit is a decoding setting stored in the weights

**Date:** 2026-11-18 · **Protocol:** `phase8/protocol.md` (pre-registered 2026-11-14, thresholds
unchanged) · **Model:** Qwen3.5-0.8B only · **Benchmark:** the 500 MATH-500 problems of phase 3c,
one sample per problem, every comparison paired over the same 500 problems.

**Corrected 2026-11-24 (ISSUES #19):** every accuracy was re-scored with the corrected answer
checker; the arm table gives first-reported -> re-scored for every arm.

**Note added 2026-11-25 (ISSUES #31).** Which line of an unboxed answer the checker reads moves
some numbers in this file; nothing was re-scored. B-2 reads −4.0 [−8.6, 0.6], a hit exactly on its
bar, and would read −4.2 [−8.6, 0.4] under both other readings — INCONCLUSIVE rather than a hit.
The exploratory C1-against-B2 comparison reads −4.8 [−9.2, −0.6] (FALLS SHORT) and would read
−1.8 [−6.2, 2.4] or −1.6 [−5.8, 2.6] (MATCHES); C1 against E1 reads +1.4 [−2.6, 5.2] and would
read +3.0 [−0.8, 6.8] or +4.0 [0.2, 7.8], still MATCHES by the rule, the interval excluding zero
only under the second; C1's own score would read 39.6% or 40.8% instead of 36.6%. Every other
registered verdict word in this file keeps its word under both other readings; C-3 keeps its miss
by 0.2 points under the first (the gap after forcing reads 11.8 and 12.6 against the bar of 11.6),
and the scorecard would then read five hits, four misses and one inconclusive.

## The answer in four sentences

1. **The blind single-row edit is a boost to the stop-thinking token and nothing else.** Adding a
   constant +9 to that token's score at generation time, with the weights untouched, scores 36.6%
   against the edit's 35.2%.
2. **Only the output half of the edit does anything.** Applied to the output layer alone it scores
   33.8% (the full edit: 35.2%). Applied to the input embedding alone it scores 9.0% (stock: 8.4%).
3. **No edit is needed to get there, but the published edit goes further.** Cutting stock's
   thinking off at 30,720 tokens and telling it to answer scores 35.4%. Turning thinking off and
   keeping our usual sampling settings scores 41.4%, 6.2 points above the blind edit. Against the
   original authors' published edit (48.6%) the two graders disagree: under `rescore.py` that
   edit is 7.2 points ahead (95% interval 2.8 to 11.8); under math-verify the two are not
   distinguishable.
4. **The original authors' edit is not explained this way.** It stays ahead of the blind edit even
   when both are forced to answer (14.4 points; 13.4 before forcing), and the +9 boost falls
   short of it (by 12.0 points under `rescore.py`, 5.8 under math-verify).

Sentences 1, 2 and 4 hold under both graders (the repo's `rescore.py` and math-verify). So does
the part of sentence 3 that concerns blind edits: no blind edit reaches the unedited model with
thinking switched off (the best blind edit, MiMo-v2.6-flash in phase 4b, scores 36.2%). The
published edit does: 48.6% against 41.4%, 7.2 points (95% interval 2.8 to 11.8) under
`rescore.py`, not distinguishable under math-verify. Until the checker's second correction of
2026-11-24 (ISSUES #19) this file said that no edited model measured in this project beats the
unedited model with thinking switched off. As first reported (old checker), the same sentences
read: 28.2% against 27.8%; 26.4% and 5.8% (stock 6.0%); 29.6% and 38.8% against 39.4%; "about
12 points".

## Every arm

"Finished" is the share of the 500 generations that ended on their own instead of hitting the
32,768-token cap. The two accuracy columns are the two graders; they agree on only 81.6% to 90.6%
of rows depending on the arm (Part A), so the protocol requires both. In the `rescore.py` column
each arm is given as first reported -> re-scored 2026-11-24 (ISSUES #19).

| Arm | What it is | `rescore.py` (first reported -> re-scored) | math-verify | Finished |
|---|---|---|---|---|
| S | stock, thinking on (carried from phase 3c) | 6.0% -> **8.4%** | 21.2% | 9.4% |
| O | original authors' published edit (carried) | 39.4% -> **48.6%** | 44.2% | 73.6% |
| E1 | blind edit: `</think>` row × 1.5 (carried) | 27.8% -> **35.2%** | 34.8% | 58.0% |
| B1 | stock, thinking off, model-card sampling | 28.8% -> **31.8%** | 30.8% | 99.4% |
| B2 | stock, thinking off, our usual sampling | 38.8% -> **41.4%** | 43.2% | 91.6% |
| B3 | E1, thinking off, model-card sampling | 24.2% -> **27.8%** | 29.2% | 100.0% |
| C1 | stock, thinking on, constant +9 on `</think>` | 28.2% -> **36.6%** | 38.4% | 63.6% |
| C2-S | stock, forced to answer at 30,720 tokens | 29.6% -> **35.4%** | 35.0% | 38.2% |
| C2-O | O, forced the same way | 41.0% -> **50.4%** | 45.0% | 75.8% |
| C2-E1 | E1, forced the same way | 28.6% -> **36.0%** | 35.6% | 58.6% |
| D-out | untied copy, output-layer row × 1.5 only | 26.4% -> **33.8%** | 34.6% | 56.6% |
| D-in | untied copy, input-embedding row × 1.5 only | 5.8% -> **9.0%** | 20.6% | 10.2% |

## Prediction scorecard

Decision rule: MATCHES means the two arms are within 4.0 points. FALLS SHORT / EXCEEDS means more
than 4.0 points apart with the 95% interval excluding zero. A prediction is a HIT only if it holds
under both graders.

| # | Prediction | `rescore.py` | math-verify | Result |
|---|---|---|---|---|
| A1 | E1's finished answers are ≥ 10 points less accurate than O's | 60.0% vs 64.7% (4.7 apart) | not computed | **MISS** |
| A2 | E1's finished generations are shorter than O's | 3,929 vs 16,297 tokens | same data | **HIT** |
| A3 | ≥ 2 points of E1's score come from cut-off generations | 0.0 points | not computed | **MISS** |
| B-1 | B1 lands in 40–52% | 31.8% | 30.8% | **MISS** |
| B-2 | B3 within ±4 points of B1 | −4.0 [−8.6, 0.6] MATCHES (at the bar) | −1.6 [−6.0, 3.0] MATCHES | **HIT** (at the bar under `rescore.py`) |
| C-1 | C1 MATCHES E1 | +1.4 [−2.6, 5.2] MATCHES | +3.6 [−0.6, 7.8] MATCHES | **HIT** |
| C-2 | forced S ≥ 16.0% | 35.4% | 35.0% | **HIT** |
| C-3 | forcing shrinks the O − E1 gap below 11.6 | 14.4 (was 13.4) | 9.4 (was 9.4) | **MISS** |
| D-1 | D-out MATCHES E1 | −1.4 [−5.6, 2.8] MATCHES | −0.2 [−4.6, 4.2] MATCHES | **HIT** |
| D-2 | D-in MATCHES S | +0.6 [−2.0, 3.2] MATCHES | −0.6 [−4.4, 3.2] MATCHES | **HIT** |

Six hits, four misses. B-2 became a hit with the checker's second correction of 2026-11-24
(ISSUES #19): B3 sits exactly at the bar under `rescore.py` (4.0 points below B1) and 1.6 points
below it under math-verify, so the prediction holds under both graders. As first reported, and
after the first correction, B-2 was a miss and the count was five hits, five misses. Three of
the hits are the mechanism claims (C-1, D-1, D-2). The misses are where the author's
expectations about answer quality and about the outside thinking-off numbers were wrong.

## Part A — re-analysis of the saved runs (no GPU)

Full numbers in `phase8/part_a.json`.

- **The two graders agree on 81.6% to 90.6% of rows** (stock 82.8, published edit 82.4, mimo
  90.0, GLM-Flash 90.6, random control 81.6; the first version of this file said 82–88%). For the
  published edit the two now read 48.6 against 44.2 on the same generations and agree on 82.4%
  of its rows: math-verify, given a whole answer section, takes the last `$…$` span and so
  misses an answer written as `The value of $x$ is 83` (ISSUES #19). The widest disagreements are
  on the arms that rarely finish, stock (8.4 against 21.2) and the random control (3.8 against
  19.8): math-verify finds a gradable answer inside thinking text that `rescore.py` refuses to
  read. The size of the blind edit's gain over stock therefore depends on the grader: +26.8
  points [22.6, 31.0] or +13.6 points [9.0, 18.2]. Both exclude zero.
- **The blind edit's finished answers are not much worse than the original edit's** (60.0% vs
  64.7% correct among finished). A1 predicted a 10-point gap. The fixed reading "the blind edit buys
  termination at the cost of answer quality" does not apply.
- **GLM's three-row edit (E2) is not distinguishable from the single-row edit (E1):** −3.6 points
  [−7.8, 0.6] and −3.8 [−8.4, 0.6]. Phase 4's ranking of the blind editors is inside the noise.
- **Nothing is scored from cut-off thinking** under `rescore.py`: 0.0 points in every arm.

## Part B — thinking off

- B1 (model-card settings) scores 31.8% / 30.8%, not the 44–50% that Toledo and shapor report.
  Why is **not established here**. Candidates not tested: their grader, a benchmark question that
  asks for a boxed answer (ours does not), a different serving stack.
- With thinking off the model still deliberates in the open: under the card settings the median
  answer is 9,254 tokens long, and 36.8% of outputs write their own `</think>` although the
  question's chat text already closed thinking. Accuracy falls with length (51 of the 90 answers
  under 1,000 tokens are right, 18 of the 132 answers over 16,000).
- **The sampling settings matter more than the edits do.** Thinking off with our usual settings
  (B2, temperature 0.6) beats thinking off with the card settings (B1, temperature 1.0) by 9.6
  points [5.0, 14.4] (12.4 under math-verify).
- Fixed reading from the protocol: B1 does not beat E1 by the decision rule (it MATCHES, −3.4 and
  −4.0), and B1 FALLS SHORT of O. So the pre-registered restatement is the weaker one: **the blind
  edit repairs thinking mode; it does not beat turning thinking off.**
- Not pre-registered, exploratory: B2 EXCEEDS E1 (+6.2 [1.8, 10.6]; +8.4 [3.8, 13.0]). Against
  the published edit (O) the two graders disagree: under `rescore.py` B2 FALLS SHORT (−7.2
  [−11.8, −2.8]; 50 problems right only with thinking off, 86 only under the published edit),
  under math-verify it MATCHES (−1.0 [−6.2, 4.2]). **The published edit beats the unedited model
  with thinking switched off and our usual sampling; no blind edit does (the best blind edit
  scores 36.2%, 5.2 points below it).**
- B3: the edit does not clearly hurt thinking-off mode (−4.0 [−8.6, 0.6], exactly at the decision
  rule's bar; −1.6 under math-verify). (Under the checker's other line rules −4.2 [−8.6, 0.4]; ISSUES #31.)

## Part C — decoding-only controls on stock

**C1, constant bias on `</think>`.** Integrity probes passed: bias 0 reproduced stock byte for
byte on the 4 probe problems, bias +30 closed thinking at the first generated token on all 4.
All 7 budgeted practice passes were used (64 GSM8K practice problems, cap 8,192, hypotheses in
`phase8/c1_session/ledger.md`):

| Bias | Practice accuracy | Thinking closed | Hit the cap |
|---|---|---|---|
| +2 | 26.6% | 35.9% | 65.6% |
| +4 | 31.2% | 46.9% | 60.9% |
| +6 | 43.8% | 70.3% | 43.8% |
| +7 | 46.9% | 82.8% | 35.9% |
| +8 | 48.4% | 95.3% | 23.4% |
| **+9** | **57.8%** | 100.0% | 4.7% |
| +10 | 37.5% | 100.0% | 14.1% |

The selection rule (highest practice accuracy) picked +9. One practice pass has a standard error
of about 6 points and each pass uses its own seed, so the jump from +9 to +10 is partly noise.
Practice numbers are as recorded with the old checker; the generations were not saved, so they
cannot be re-scored. Final at +9 on MATH-500: **36.6%** (re-scored 2026-11-24, ISSUES #19; 28.2% as
first reported), MATCHES E1 under both graders. Exploratory: it FALLS SHORT of O (−12.0 [−16.4,
−7.8]; −5.8 [−10.8, −1.0]).

**Where thinking ends (added 2026-11-24).** The +9 bias closes thinking at the very first
generated token in 298 of the 500 problems, and 338 of the 500 generations emit `</think>` again
inside the answer. Accuracy is the same whether any thinking happened: the same share of each
group is right (109 of the 298 with none, 74 of the 202 with some). The single-row blind edits
do the same thing more mildly: the GLM-Flash edit starts with `</think>` in 239 of 500
generations, the mimo edit in 152 of 500, the phase-10 edit in 150 of 500; the published edit
in 0 of 500. So "thinking closed 100%" for C1
means "thinking removed in three of five problems", and the C1-against-E1 match is supporting
evidence for the mechanism claim; the input/output split (Part D) carries it.

**C2, forced answers.** Generations that had not closed thinking within 30,720 tokens were
truncated there, given Qwen3's budget-exceeded cue, and allowed 2,048 more tokens.

| Source run | Rows that needed forcing | Before | After (`rescore.py`) |
|---|---|---|---|
| S (stock) | 452 of 500 | 8.4% | 35.4% |
| O (original edit) | 99 of 500 | 48.6% | 50.4% |
| E1 (blind edit) | 12 of 500 | 35.2% | 36.0% |

- Forced stock MATCHES the blind edit (+0.2 [−3.6, 4.0]; +0.2 [−4.2, 4.6]; exploratory comparison).
  Stock already had the answers; it was not stopping to give them.
- E1 almost never needed forcing, yet only 58% of its generations finish. Its remaining failures
  are loops **after** thinking has closed, which neither the edit nor forcing reaches.
- Forcing does not close the gap between the original edit and the blind edit (14.4 points after,
  13.4 before). The original edit's advantage is not just "it stops thinking".
- Disclosed risk, measured: re-encoding the saved text gave a different token count for 16.2% (S),
  79.0% (O) and 61.2% (E1) of generations. The typical difference is exactly one token, in the
  direction of the end-of-text token that is counted in `n_tokens` but absent from saved text
  (343 of O's 395 mismatches, 287 of E1's 306, 64 of S's 81). Largest difference: 1,205 tokens on
  one stock row.

## Part D — the edit's two halves

The untied copy with nothing scaled (U0) reproduced stock byte for byte on the 4 probe problems,
so Part D is valid. Each built model was checked to differ from stock in the intended row only.

- **D-out** (output layer row × 1.5): 33.8% / 34.6%, MATCHES E1.
- **D-in** (input embedding row × 1.5): 9.0% / 20.6%, MATCHES S.

## Combined reading (fixed in the protocol before any run)

C1 vs E1 is MATCHES and D-out vs E1 is MATCHES, so the pre-registered sentence applies verbatim:

> The blind edit is a decoding-time boost to the stop-thinking token, stored in the weights. The
> "three editors converged" result is restated as convergence on a decoding fix.

## What this changes in the project's record

- Phases 4, 9 and 10 measured editors finding a generation setting, reachable in one line of
  sampling config with the weights untouched.
- The phase 4 headline becomes: the blind edit repairs thinking mode; it does not beat turning
  thinking off. With our usual sampling settings, turning thinking off beats it by 6.2 points
  [1.8, 10.6].
- The phase 10 claim that the hint buys a "deeper" edit does not survive (ISSUES #23). What is
  measured is a quick fix against a deeper edit under different rules, practice data and editor
  models; the experiment that would isolate the hint — a blind editor under the original rules,
  with in-distribution practice — has not been run. (Run since, as phase 14 on 2026-11-25/26: the
  blind editor scored 9.0%, not distinguishable from stock's 8.4% and 39.6 points below the
  published edit; `phase14/results.md`.) What stands is narrower: the original edit
  beats the +9 stop-token control here (C1 is 12.0 points [−16.4, −7.8] below it) and, under
  `rescore.py`, thinking off as well (B2 is 7.2 points [−11.8, −2.8] below it; not
  distinguishable under math-verify, −1.0 [−6.2, 4.2]).
- The grader caveat sits on the arms that rarely finish (stock 8.4% vs 21.2%; the
  input-embedding-only edit 9.0% vs 20.6%; Part A's random control 3.8% vs 19.8%). On the arms
  that finish the two graders are farthest apart on the published edit (48.6% vs 44.2%; forced
  to answer, 50.4% vs 45.0%). (First version: "a grader caveat of up to 15 points on arms that
  rarely finish".)

## Deviations and limits (also in ISSUES.md #16–20)

- The second grader ran in its own environment (`~/wsp8-grader-venv`, Python 3.10, math-verify
  0.9.0), not in the vLLM environment the protocol names, to leave that environment untouched.
- math-verify times out on 1–2 rows per run, and which rows varies between reruns. Part A counts
  them as ungradable; `verdicts.py` counts them as wrong. Effect: at most 0.4 points per arm.
- The first launch of the GPU chain collided with another session's run on the same card and was
  stopped during the probes. No finished arm comes from that launch; the probes were redone.
- Five comparisons were added after the arm accuracies were known (C2-S vs E1, B2 vs E1, B2 vs O,
  C1 vs O, C1 vs B2). They are marked exploratory above and settle no prediction.
- One sample per problem, one seed. A single arm has a standard error of about 2 points.
- The practice set for C1 is GSM8K at cap 8,192; the final is MATH-500 at cap 32,768, as in phase 4.
- Re-scoring of 2026-11-24 (ISSUES #19): the repository's answer checker had been discarding
  correct answers written in a sentence. It was corrected twice that day (the second time after
  a test author found that `The value of $x$ is 83` was still read as `x`), and every arm here
  was re-scored from its saved generations (`phase8/rescore_arms.py`). The first correction left
  the ten prediction verdicts as first reported; the second turned B-2 from a miss into a hit.
- Run-validity checks were added to the evaluation scripts (ISSUES #20) after a corrupted run in
  phase 7 had been read as model behaviour. `phase8/run_vllm8.py` now takes the GPU lock first,
  can bound the requests in flight (`--max-seqs`, off by default), checks each finished run with
  `phase3c/runcheck.py`, and refuses to overwrite an existing result. The runs in this file were
  made before these checks existed.

## Reproduce

Inside WSL, from the repo root. Arms that already have a JSON are skipped.

```bash
~/wsp8-grader-venv/bin/python phase8/analyze_saved.py
```

The three untied models (each command also appears in the header of `build_untied.py`):

```bash
~/wsp3c-venv/bin/python phase8/build_untied.py ~/wsp3c/stock ~/wsp8/u0
```

```bash
~/wsp3c-venv/bin/python phase8/build_untied.py ~/wsp3c/stock ~/wsp8/d_out --out-scale 1.5
```

```bash
~/wsp3c-venv/bin/python phase8/build_untied.py ~/wsp3c/stock ~/wsp8/d_in --in-scale 1.5
```

Probes, Part B, the three fixed practice passes, C2 and Part D:

```bash
bash phase8/run_chain.sh
```

The four free practice passes (+8, +10, +7, +9, in that order) were run one at a time from
`phase8/c1_session` with the command in the header of `eval_pass8.py`; their hypotheses are rows
4–7 of `c1_session/ledger.md`. Then the final:

```bash
cd phase8 && ~/wsp3c-venv/bin/python run_vllm8.py --model ~/wsp3c/stock --name c1_final --logit-bias 9
```

```bash
~/wsp8-grader-venv/bin/python phase8/verdicts.py
```

Re-scoring (ISSUES #19), to run before either analysis script:

```bash
uv run python phase8/rescore_arms.py
```

It re-scores every phase-8 arm's JSON in place from its saved generations (arguments name
single arms); the five carried runs are re-scored the same way by
`uv run python phase3c/rescore.py <run> [<run> ...]`.

Outputs: `part_a.json`, `probes.json`, `<arm>.json` and `<arm>_texts.json.gz` for each arm,
`c1_session/` (ledger, counter, one JSON per practice pass), `verdicts.json`, `logs/`.
