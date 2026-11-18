# Phase 8 results — the missing baselines: the blind edit is a decoding setting stored in the weights

**Date:** 2026-11-18 · **Protocol:** `phase8/protocol.md` (pre-registered 2026-11-14, thresholds
unchanged) · **Model:** Qwen3.5-0.8B only · **Benchmark:** the 500 MATH-500 problems of phase 3c,
one sample per problem, every comparison paired over the same 500 problems.

## The answer in four sentences

1. **The blind single-row edit is a boost to the stop-thinking token and nothing else.** Adding a
   constant +9 to that token's score at generation time, with the weights untouched, scores 28.2%
   against the edit's 27.8%.
2. **Only the output half of the edit does anything.** Applied to the output layer alone it scores
   26.4% (the full edit: 27.8%). Applied to the input embedding alone it scores 5.8% (stock: 6.0%).
3. **No edit is needed to get there.** Cutting stock's thinking off at 30,720 tokens and telling it
   to answer scores 29.6%. Turning thinking off and keeping our usual sampling settings scores
   38.8%, level with the original authors' published edit (39.4%) and 11 points above the blind edit.
4. **The original authors' edit is not explained this way.** It stays about 12 points ahead of the
   blind edit even when both are forced to answer, and the +9 boost falls short of it.

All four hold under both graders (the repo's `rescore.py` and math-verify).

## Every arm

"Finished" is the share of the 500 generations that ended on their own instead of hitting the
32,768-token cap. The two accuracy columns are the two graders; they agree on only 82–88% of rows
(Part A), so the protocol requires both.

| Arm | What it is | `rescore.py` | math-verify | Finished |
|---|---|---|---|---|
| S | stock, thinking on (carried from phase 3c) | 6.0% | 21.2% | 9.4% |
| O | original authors' published edit (carried) | 39.4% | 44.2% | 73.6% |
| E1 | blind edit: `</think>` row × 1.5 (carried) | 27.8% | 34.8% | 58.0% |
| B1 | stock, thinking off, model-card sampling | 28.8% | 30.8% | 99.4% |
| B2 | stock, thinking off, our usual sampling | 38.8% | 43.2% | 91.6% |
| B3 | E1, thinking off, model-card sampling | 24.2% | 29.2% | 100.0% |
| C1 | stock, thinking on, constant +9 on `</think>` | 28.2% | 38.4% | 63.6% |
| C2-S | stock, forced to answer at 30,720 tokens | 29.6% | 35.0% | 38.2% |
| C2-O | O, forced the same way | 41.0% | 45.0% | 75.8% |
| C2-E1 | E1, forced the same way | 28.6% | 35.6% | 58.6% |
| D-out | untied copy, output-layer row × 1.5 only | 26.4% | 34.6% | 56.6% |
| D-in | untied copy, input-embedding row × 1.5 only | 5.8% | 20.6% | 10.2% |

## Prediction scorecard

Decision rule: MATCHES means the two arms are within 4.0 points. FALLS SHORT / EXCEEDS means more
than 4.0 points apart with the 95% interval excluding zero. A prediction is a HIT only if it holds
under both graders.

| # | Prediction | `rescore.py` | math-verify | Result |
|---|---|---|---|---|
| A1 | E1's finished answers are ≥ 10 points less accurate than O's | 46.9% vs 51.9% (5.0 apart) | not computed | **MISS** |
| A2 | E1's finished generations are shorter than O's | 3,929 vs 16,297 tokens | same data | **HIT** |
| A3 | ≥ 2 points of E1's score come from cut-off generations | 0.0 points | not computed | **MISS** |
| B-1 | B1 lands in 40–52% | 28.8% | 30.8% | **MISS** |
| B-2 | B3 within ±4 points of B1 | −4.6 [−9.0, 0.0] INCONCLUSIVE | −1.6 [−6.0, 3.0] MATCHES | **MISS** (holds under one grader only) |
| C-1 | C1 MATCHES E1 | +0.4 [−3.6, 4.4] MATCHES | +3.6 [−0.6, 7.8] MATCHES | **HIT** |
| C-2 | forced S ≥ 16.0% | 29.6% | 35.0% | **HIT** |
| C-3 | forcing shrinks the O − E1 gap below 11.6 | 12.4 (was 11.6) | 9.4 (was 9.4) | **MISS** |
| D-1 | D-out MATCHES E1 | −1.4 [−5.4, 2.6] MATCHES | −0.2 [−4.6, 4.2] MATCHES | **HIT** |
| D-2 | D-in MATCHES S | −0.2 [−2.4, 2.0] MATCHES | −0.6 [−4.4, 3.2] MATCHES | **HIT** |

Five hits, five misses. The hits are the mechanism claims (C-1, D-1, D-2). The misses are where
the author's expectations about answer quality and about the outside thinking-off numbers were wrong.

## Part A — re-analysis of the saved runs (no GPU)

Full numbers in `phase8/part_a.json`.

- **The graders disagree on 12–18% of rows**, mostly on unfinished generations: math-verify finds a
  gradable answer inside thinking text that `rescore.py` refuses to read. Stock is 6.0% under one
  and 21.2% under the other. The size of the blind edit's gain over stock therefore depends on the
  grader: +21.8 points [17.8, 25.8] or +13.6 points [9.0, 18.2]. Both exclude zero.
- **The blind edit's finished answers are not much worse than the original edit's** (46.9% vs
  51.9% correct among finished). A1 predicted a 10-point gap. The fixed reading "the blind edit buys
  termination at the cost of answer quality" does not apply.
- **GLM's three-row edit (E2) is not distinguishable from the single-row edit (E1):** −2.6 points
  [−6.8, 1.6] and −3.8 [−8.4, 0.6]. Phase 4's ranking of the blind editors is inside the noise.
- **Nothing is scored from cut-off thinking** under `rescore.py`: 0.0 points in every arm.

## Part B — thinking off

- B1 (model-card settings) scores 28.8% / 30.8%, not the 44–50% that Toledo and shapor report.
  Why is **not established here**. Candidates not tested: their grader, a prompt that asks for a
  boxed answer (ours does not), a different serving stack.
- With thinking off the model still deliberates in the open: under the card settings the median
  answer is 9,254 tokens long, and 36.8% of outputs write their own `</think>` although the prompt
  already closed thinking. Accuracy falls with length (46.7% on the 90 answers under 1,000 tokens,
  12.9% on the 132 answers over 16,000).
- **The sampling settings matter more than the edits do.** Thinking off with our usual settings
  (B2, temperature 0.6) beats thinking off with the card settings (B1, temperature 1.0) by 10.0
  points [5.4, 14.8] (12.4 under math-verify).
- Fixed reading from the protocol: B1 does not beat E1 by the decision rule (it MATCHES, +1.0 and
  −4.0), and B1 FALLS SHORT of O. So the pre-registered restatement is the weaker one: **the blind
  edit repairs thinking mode; it does not beat turning thinking off.**
- Not pre-registered, exploratory: B2 EXCEEDS E1 (+11.0 [6.4, 15.4]; +8.4 [3.8, 13.0]) and MATCHES
  O (−0.6 [−5.4, 4.0]; −1.0 [−6.2, 4.2]). **No edited model measured in this project beats the
  unedited model with thinking switched off and our usual sampling.**
- B3: the edit does not clearly hurt thinking-off mode (−4.6, interval touching zero; −1.6 under
  math-verify).

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
Final at +9 on MATH-500: **28.2%**, MATCHES E1 under both graders. Exploratory: it FALLS SHORT of O
(−11.2 [−15.6, −6.8]; −5.8 [−10.8, −1.0]).

**C2, forced answers.** Generations that had not closed thinking within 30,720 tokens were
truncated there, given Qwen3's budget-exceeded cue, and allowed 2,048 more tokens.

| Source run | Rows that needed forcing | Before | After (`rescore.py`) |
|---|---|---|---|
| S (stock) | 452 of 500 | 6.0% | 29.6% |
| O (original edit) | 99 of 500 | 39.4% | 41.0% |
| E1 (blind edit) | 12 of 500 | 27.8% | 28.6% |

- Forced stock MATCHES the blind edit (+1.8 [−2.2, 5.8]; +0.2 [−4.2, 4.6]; exploratory comparison).
  Stock already had the answers; it was not stopping to give them.
- E1 almost never needed forcing, yet only 58% of its generations finish. Its remaining failures
  are loops **after** thinking has closed, which neither the edit nor forcing reaches.
- Forcing does not close the gap between the original edit and the blind edit (12.4 points after,
  11.6 before). The original edit's advantage is not just "it stops thinking".
- Disclosed risk, measured: re-encoding the saved text gave a different token count for 16.2% (S),
  79.0% (O) and 61.2% (E1) of generations. The typical difference is exactly one token, in the
  direction of the end-of-text token that is counted in `n_tokens` but absent from saved text
  (343 of O's 395 mismatches, 287 of E1's 306, 64 of S's 81). Largest difference: 1,205 tokens on
  one stock row.

## Part D — the edit's two halves

The untied copy with nothing scaled (U0) reproduced stock byte for byte on the 4 probe problems,
so Part D is valid. Each built model was checked to differ from stock in the intended row only.

- **D-out** (output layer row × 1.5): 26.4% / 34.6%, MATCHES E1.
- **D-in** (input embedding row × 1.5): 5.8% / 20.6%, MATCHES S.

## Combined reading (fixed in the protocol before any run)

C1 vs E1 is MATCHES and D-out vs E1 is MATCHES, so the pre-registered sentence applies verbatim:

> The blind edit is a decoding-time boost to the stop-thinking token, stored in the weights. The
> "three editors converged" result is restated as convergence on a decoding fix.

## What this changes in the project's record

- Phases 4, 9 and 10 ("ten blind sessions converge on row 248069") measured editors finding a
  generation setting, reachable in one line of sampling config with the weights untouched.
- The phase 4 headline becomes: the blind edit repairs thinking mode, it does not beat turning
  thinking off, and with our usual sampling settings turning thinking off beats it by 11 points.
- The phase 10 claim that the hint buys a "deeper" edit survives in a narrower form: the original
  edit beats every stop-token control here by about 11 points. It does not beat thinking off (B2).
- Every accuracy in the record carries a grader caveat of up to 15 points on arms that rarely
  finish (stock: 6.0% vs 21.2%).

## Deviations and limits (also in ISSUES.md #16–18)

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

Outputs: `part_a.json`, `probes.json`, `<arm>.json` and `<arm>_texts.json.gz` for each arm,
`c1_session/` (ledger, counter, one JSON per practice pass), `verdicts.json`, `logs/`.
