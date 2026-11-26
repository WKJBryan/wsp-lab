# Phase 3c results — vLLM cross-check: the original's weight edit TRANSFERS on local hardware

**Date:** 2026-11-12 · **Engine:** vLLM 0.33.0 under WSL2 (Ubuntu 22.04, CUDA 12.8 toolkit, RTX 5090)
**Protocol:** phase3b protocol with cap restored to the original's **32,768**; temp 0.6, top-p 0.95,
top-k 20, presence penalty 1.5 (vLLM native), thinking on, 1 sample, all 500 MATH-500 problems,
identical for both arms. Wall time: **50 min (stock) + 31 min (edited)** — vs 17–24 h per arm on
the transformers fallback stack.

## Final numbers

| Arm | Accuracy | Answered | Cap-hit | Mean tokens |
|---|---|---|---|---|
| Stock `Qwen/Qwen3.5-0.8B` | **9.4%** (47/500) | 9.6% | 90.6% | 30,108 |
| Original's published edit (layers 15/19 down_proj) | **54.2%** (271/500) | 78.2% | 26.4% | 20,645 |

**Corrected 2026-11-24 (ISSUES #19).** Both accuracies were re-scored with the final answer
checker of that day (the checker was corrected twice that day; since 2026-11-26 that name belongs
to the checker of ISSUES #30, #31). As first reported they were stock 6.0%
(30/500) and published edit 39.4% (197/500); after the first correction they read 7.0% (35/500)
and 43.8% (219/500), after the second 8.4% (42/500) and 48.6% (243/500). The Answered, Cap-hit and Mean-tokens columns are unchanged.

**Re-scored 2026-11-26 (ISSUES #30, #31).** The table is the final answer checker's reading:
stock 9.4% (47/500), published edit 54.2% (271/500). The Answered column moved with this
re-score (9.8% and 83.2% under the checker of 2026-11-24), because the final checker no longer
counts a line that cannot be a conclusion; Cap-hit and Mean tokens are not readings of the
checker and did not move.

**Pre-registered verdict (phase3b thresholds): TRANSFERS.** The published edit scores **44.8 points
more than stock** (95% interval 40.4 to 49.2; 225 problems right only under the published edit,
1 only under stock) — far beyond the +10.0 bar. The share of generations that finish — ending
on their own instead of hitting the 32,768-token cap — flips from 9.4% to 73.6%. This is the
original's core claim, reproduced locally by us, on the user's own GPU.
(Corrected 2026-11-24, ISSUES #19: this paragraph said "+33.4 points, 6.6× relative" and
"Termination flips from 9% to 83% of generations"; 83.2% is the share that produced an answer, not
the share that finished. Re-scored 2026-11-26, ISSUES #30, #31: under the checker of 2026-11-24 the
difference read 40.2 points, interval 35.8 to 44.6, 204 against 3; the Answered column then
read 9.8% and 83.2%.)

Anchor comparison (corrected 2026-11-24, ISSUES #19): stock 8.4 against openwsp's 7.2, Toledo's
independent 7.8% (93.1% cap-hit) and the original authors' 7.9% on their own stack — agrees.
Published edit 48.6 against openwsp's 50.6, a third party's 51.0 and the original's 48.1: within
about two and a half points of the outside replications (2.0 points below openwsp, 2.4 below the
third party, 0.5 above the original authors' own number). Known differences between the two
set-ups: our benchmark question is the bare problem while theirs adds "Please reason step by
step, and put your final answer within `\boxed{}`"; we take one sample per problem, they four;
and the two graders do not read every answer the same way — they agree on 82% of the published
edit's rows (48.6 against 44.2 on the same generations): math-verify, given a whole answer
section, takes the last `$…$` span and so misses an answer such as "The value of `$x$` is 83"
(ISSUES #19). The calibration runs on their exact protocol (2026-11-24/25; next subsection)
show how much these differences carry: with their benchmark question and their grader, and one
sample per problem as in our own runs, this harness reads 51.0% for the published edit and 7.8%
for stock, reproducing the outside numbers. So the gap to the outside replications is closed,
and the repository's matcher — stricter than math-verify on the same boxed answers — accounts
for most of the remaining difference between 48.6 and the outside 50.6 to 51.0 (ISSUES #30).
The earlier text put the shortfall at "~9 points shy" and called the gap robust because
"extraction micro-choices moved each arm by ≤1.6 points"; that was wrong — the re-scoring alone
moved the gap from +33.4 to +40.2.

**Anchor comparison under the final answer checker (2026-11-26, ISSUES #30, #31).** On our
protocol the two arms read stock 9.4% and published edit 54.2% (the table above; 48.6% under the
checker of 2026-11-24) — the published edit is now above the outside readings of the same edit
(the original authors' 48.1, openwsp's 50.6 and a third party's 51.0), not within two and a half
points of them — and the two graders agree on 84.0% of its rows (54.2 against 44.2 on the same
generations). The conclusion that the harness reproduces the outside numbers stands on the
calibration runs, which use the original authors' protocol (next subsection): there this stack
reads 8.8% for stock and 51.8% for the published edit under math-verify, their grader (51.0%
before the third rule), and the repository's matcher on the same boxed answers 51.6% (46.6%
before), 0.2 points below math-verify. The two protocols read the same edit a few points apart:
54.2% on ours, 51.8% on theirs. So the paragraph above is history: the gap it discussed is gone.

### Calibration on the original authors' protocol (2026-11-24/25)

Three calibration runs test whether the remaining distance to the outside numbers comes from
our set-up: `phase3c/ref_stock.json`, `ref_orig.json` and `ref_nothink_card.json`, each with its
generations (`_texts.json.gz`) and its run log (`phase3c/logs/ref_*.log`); each also carries a
`_rescored.json`, the repository's checker applied to the whole answer section (8.8%, 54.4% and
46.4% under the final answer checker; 7.4%, 49.0% and 41.8% under the checker of 2026-11-24).
They are graded by `phase3c/grade_reference.py`; its log is `phase3c/logs/grade_reference.log`,
re-run on 2026-11-26 under the final answer checker, which reads a box at any depth and passes
over blank ones (ISSUES #30, #31). The readings of 2026-11-24/25 are kept below the table.

Protocol: the original authors' benchmark question ("Problem: … Please reason step by step, and
put your final answer within `\boxed{}`."), thinking on, temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5, at most 32,768 tokens, one sample, the same 500 problems. The third run
has thinking switched off under the model card's non-thinking settings (temperature 1.0, top_p
1.00, top_k 20, presence penalty 2.0). Graded as the outside replications grade — math-verify on
the last `\boxed{}` answer — and, on the same boxed answers, with the repository's own matcher.

| Version | an answer in `\boxed{}` | repository matcher | math-verify | outside (math-verify) | at the token cap |
|---|---|---|---|---|---|
| Stock, thinking on | 44 of 500 | 8.8% (44 of 500) | 8.8% (44 of 500) | 7.2% (openwsp) | 93.2% |
| Published edit | 320 of 500 | 51.6% (258 of 500) | 51.8% (259 of 500) | 50.6% (openwsp), 51.0% (a third party), 48.1% (the original authors) | 29.0% |
| Stock, thinking off, card settings | 496 of 500 | 46.0% (230 of 500) | 47.0% (235 of 500) | 46.8% (openwsp) | 0.6% (99.4% finished) |

Mean tokens, in the same order: 30,991; 21,796; 9,067. All three runs pass both validity checks.

As read on 2026-11-24/25, under the checker of 2026-11-24 (its box was read one brace level deep,
and a blank box counted): an answer in `\boxed{}` in 123, 331 and 487 of 500; repository matcher
7.4% (37 of 500), 46.6% (233 of 500) and 41.2% (206 of 500); math-verify 7.8% (39 of 500),
51.0% (255 of 500) and 46.2% (231 of 500); the matcher 4.4 and 5.0 points stricter than
math-verify.

Paired, on the same boxed answers, over the same 500 problems:

- Repository matcher: published edit against stock +42.8 points (95% interval 38.4 to 47.2; 215
  problems right only under the edit, 1 only under stock; p = 4.1e-63).
- Repository matcher: published edit against stock with thinking off +5.6 points (interval 1.2
  to 10.0; 78 only under the edit, 50 only with thinking off; p = 0.017).
- math-verify: published edit against stock +43.0 points (interval 38.6 to 47.4; 216 problems
  right only under the edit, 1 only under stock; p = 2.1e-63).
- math-verify: published edit against stock with thinking off +4.8 points (interval 0.4 to 9.4;
  76 only under the edit, 52 only with thinking off; p = 0.042).
- As read on 2026-11-24/25: repository matcher +39.2 (interval 35.0 to 43.6; 197 against 1) and
  +5.4 (1.0 to 9.6; 76 against 49; p = 0.020); math-verify +43.2 (38.8 to 47.6; 217 against 1)
  and +4.8 (0.4 to 9.2; 76 against 52; p = 0.042).

Agreement of the two graders on the same boxed answers: stock 44 of 44; published edit 319 of
320 — the other one is an answer that math-verify accepts and the matcher rejects, never the
reverse; thinking off 491 of 496 — five such answers, never the reverse. So the matcher is
stricter than math-verify by 0.2 points on the published edit and 1.0 on stock with thinking off,
and not at all on stock with thinking on (as read on 2026-11-24/25: 498, 478 and 475 of 500 in
agreement, the matcher 4.4 and 5.0 points stricter).

What those answers were under the checker of 2026-11-24: gold answers in LaTeX shorthand the matcher did not expand (`\frac43`,
`\frac 59`, `11\sqrt2`, `\dfrac{33}{100}`); a degree sign or a unit in the gold (`90^\circ` for
90; `864 \mbox{ inches}^2` for 864); a multiple-choice letter in parentheses (`\text{(C)}` for
C); a variable in front of the gold (`x=5`, `x \in [-2,7]`); thousands separators (`58,500`);
a tuple with a fraction or decimal inside (`(\frac{3}{2}, -13)`); a base written `4210_{5}`;
braces around one number (`\{-4\}`); in the thinking-off run also a sum written in reverse
order, a unit appended to a correct value, and the capital in "East". Two or three are
debatable (`40` for `40_9`).

Under the final answer checker (2026-11-26) the matcher also accepts those formatting
equivalents (ISSUES #30), and six answers remain that math-verify accepts and it rejects: digits
written without their base (`40` for `40_9`, `4343` for `4343_6`); factors in another order
(`\sqrt{2}\cdot 11` for `11\sqrt2`; a polynomial written in the other direction); a compound unit
(`12\pi` against `12\pi \text{ inches per second}`); and a matrix written with `\begin{pmatrix}`.

Reading:

1. **The harness reproduces the outside numbers under the outside protocol and grader.**
   Published edit 51.8% against 50.6, 51.0 and 48.1 outside; stock 8.8% against 7.2%; stock with
   thinking off 47.0% against 46.8%. The gap these runs were meant to explain is closed, and its
   source was the answer checker (ISSUES #19, #30, #31). (As read on 2026-11-24/25: 51.0%, 7.8%
   and 46.2%.)
2. **The repository matcher is at most a point stricter than math-verify on the same boxed
   answers, and never the reverse** (4 to 5 points stricter until the third rule). Every score
   in this repository is on that scale, and every comparison inside the repository shares the
   matcher (ISSUES #30, closed 2026-11-26).
3. **Under this protocol the published edit beats the unedited model with thinking switched off
   under both graders (+5.6 and +4.8 points)**, where our own protocol could not separate them
   under math-verify applied to the whole answer section.

## The full forensic arc (for anyone reproducing)

1. **Phase 3a (greedy, 6K cap): VOID** — a void regime (ISSUES #8): greedy decoding, which is
   not the original's sampling. (Corrected 2026-11-24, ISSUES #26: this said "decoding prohibited
   by the model card"; the sentence we attributed to the card is not on it as published.)
2. **Phase 3b (sampled on transformers/Windows fallback kernels): VOID** — model non-terminates
   under every decoding that stack can produce (ISSUES #9); math-verify also broken on-platform
   (replaced by `phase3b/scorer.py`).
3. **Engine diagnosis proven by probe:** same 4 problems — 0/8 terminate on the fallback stack,
   3/4 on vLLM.
4. **Phase 3c (vLLM/WSL2, proper kernels, native presence penalty, full 32K budget): VALID.**
   Setup additions required: CUDA toolkit 12.8 + ninja/build-essential in WSL for flashinfer JIT.
5. **Extraction adaptation (ISSUES #10):** this model answers in prose ("**Answer:** X",
   "The answer is X") — 2 boxed answers in 1,000 generations. Boxed-only scoring scored
   terminations as wrong (0.0%/0.4%). Final extractor (boxed → answer-marker lines post-`</think>`
   → bold/$-span/LaTeX/last-number-or-word fallbacks) was iterated 4 times against output-audit
   (never against golds), applied identically to both arms from saved generations. The extractor of
   that iteration lost 2 to 9 points per arm that finishes (ISSUES #19); the checker was
   corrected twice on 2026-11-24, given a third rule on 2026-11-26, and all saved runs
   re-scored after each. (It replaces the sentence
   "Remaining extraction residue: ≤~2 points per arm".)
6. **Checker correction (ISSUES #19), made in two steps on 2026-11-24:** `phase3c/rescore.py` had
   a defect: it removed a sentence's final period and then stripped the closing `$` of a math span,
   so a correct answer written as a sentence was lost ("... is $-2 + 7i$." was scored as "7"; a
   degree-9 polynomial as "1"). The first correction reads a `$math$` span that ends the answer
   before any `$` is stripped, and removes a stray `</think>` token inside the answer. A test
   author then found that the checker still misread a whole family of correct answers: when the
   model writes "The value of `$x$` is **83**", it took the `$x$` and scored "x". It had also begun
   to misread a value followed by a unit (`$\frac{16}{27}$ gallons` became
   `\frac{16}{27}gallons`), and a money answer written `$\$10$` could not match a gold answer of
   "10". What the checker of 2026-11-24 did (identically for every arm; it never looks at the gold
   answer):
   - a `$math$` span that ends the answer is the answer;
   - a span followed by nothing but a linking word ("is", "is equal to", "equals"; up to four
     words may stand before it) and a plain number or fraction states that value;
   - any other span is the answer (`$\frac{13}{15}$ of the pie` stays the fraction); an ordinal
     span ("the `$12^{th}$` grade") is its number;
   - an escaped dollar sign is a dollar sign.

   Measured answer by answer on the 20,218 saved answers of phases 3c (the fifteen valid
   500-problem runs), 8, 11 and 5, under four rules. Right answers: as first committed 4,681
   (the first-reported scores); first re-score of 2026-11-23/24 5,226; first correction as last
   refined 5,275; final answer checker of that day 5,795.
   - The checker of 2026-11-24 against the rule as first committed: **1,125 answers turn from wrong to
     right and 11 from right to wrong.** Four of the 11 were accidental hits of the old rule (it
     took a digit out of text that was not an answer). Seven are misreads that remained in the
     final answer checker of that day: a sentence inside display math; a value offered in two equal forms, the
     second taken (two answers); "There are `$76^\circ$` degrees in `$\angle B$`."; "There are
     7 such positive integers `$n$`."; "There are 4 possible values for `$x$`."; one looping,
     cut-off text. That checker was frozen at that point and these seven are on record (ISSUES #19): each
     is one answer of a 500-problem run, 0.2 points.
   - The checker of 2026-11-24 against the first re-score: **570 right, 1 wrong.** The one is the "4.76"
     false positive that rule let through (a model wrote "4.76" after "`$\frac{9}{2}$`
     expressed as a decimal is"; the checker took the fraction and scored it right against a
     gold of 4.5). The independent grader (math-verify), given only the last line of each of
     the 570 answers, agrees with 566.
   - The checker of 2026-11-24 against the first correction as last refined: **520 right, none wrong.**
   - A third limit of the checker was measured on 2026-11-25 without changing it (ISSUES #31):
     which line of an unboxed answer it reads — read the section's last line instead, and 284 of
     the 22,718 saved answers turn right and 286 wrong. The published edit itself then reads
     47.6% (48.4% under reading C of that measurement), so its distances from the outside replications above
     can read up to 3.4 points; it stays 38.8 to 40.2 points above stock with thinking on.
   - Third rule, 2026-11-26 (ISSUES #30, #31), which settles both limits above. A complete
     `\boxed{}` anywhere in the answer is read first, at any depth, blank and unclosed boxes
     passed over; without a box the answer is right if the last line carrying an answer marker
     is right or, else, the section's last line when it can be a conclusion; and the matcher also
     accepts the formatting equivalents math-verify accepts, only ever adding matches. Measured
     answer by answer on the 23,718 saved answers (the 22,718 above and the two phase-14 finals):
     943 answers turn from wrong to right and 21 from right to wrong; 6,840 right before, 7,762
     right now. Every saved run was re-scored again; the tables above are the final reading.

   139 unit tests (`phase3c/test_rescore.py`) and 104 for the matcher (`phase3b/test_scorer.py`)
   pass. Every saved run was re-scored with the final
   answer checker, and every re-scored file records the checker version that produced it. **Reproduce**
   (no GPU, no gunzip needed): `uv run python phase3c/rescore.py <run> [<run> ...]` reads the
   gzipped generations directly; `<run>` is the file stem, e.g. `uv run python
   phase3c/rescore.py p3c_stock p3c_c3_original`. The calibration runs (subsection "Calibration
   on the original authors' protocol") are graded as the outside replications grade them, with
   `uv run python phase3c/grade_reference.py <run.json>` (the grader environment).

## What this closes

- The user's challenge ("why not the same or better results?") is answered as far as the pipeline
  goes: a reproducible local run — **wrong engine → right engine; wrong scorer → right scorer;
  and the original's claim lands: 9.4 → 54.2%** (re-scored 2026-11-26; 8.4 → 48.6% under the
  checker of 2026-11-24; 6.0 → 39.4% as first reported). On our protocol the published edit reads
  54.2% (48.6% under the checker of 2026-11-24), above the outside readings of the same edit (the
  original authors' 48.1, openwsp's 50.6 and a third party's 51.0) rather than within two and a
  half points of them. The conclusion that the harness reproduces the outside numbers stands on
  the calibration runs, which use the original authors' protocol (2026-11-24/25): there our
  stack reads 51.8% for the published edit under math-verify, their grader (51.0% before the
  third rule), against their 48.1%, and 8.8% for stock against 7.2%; the repository's matcher on
  the same boxed answers reads 51.6% (46.6% before). The two protocols read the same edit a few
  points apart. So the gap to the outside numbers is closed; its source was the answer checker
  (ISSUES #19, #30, #31).
  (Corrected 2026-11-24, ISSUES #19: this bullet said the challenge was "now fully answered";
  after the first checker correction it put the remaining shortfall at "the last 4 to 7 points",
  and until the calibration runs at "the last two and a half points or so".)
- Weight-space editing stopping thinking-loops is confirmed **on third-party-independent
  hardware** (this repo) in addition to openwsp/Khaliq/Noble.
- Still unclaimed when this phase closed: the no-hint full-budget run. It has been run since
  (phase 10) — the first run we know of that offered a blind editor 50 passes, and no claim about
  the whole field. (Corrected 2026-11-24, ISSUES #23: this bullet said the run was "still unclaimed anywhere,
  including here" and called the Phase-3a blind NULL "one of the only no-hint data points in
  existence".)
