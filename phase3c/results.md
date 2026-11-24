# Phase 3c results — vLLM cross-check: the original's weight edit TRANSFERS on local hardware

**Date:** 2026-11-12 · **Engine:** vLLM 0.33.0 under WSL2 (Ubuntu 22.04, CUDA 12.8 toolkit, RTX 5090)
**Protocol:** phase3b protocol with cap restored to the original's **32,768**; temp 0.6, top-p 0.95,
top-k 20, presence penalty 1.5 (vLLM native), thinking on, 1 sample, all 500 MATH-500 problems,
identical for both arms. Wall time: **50 min (stock) + 31 min (edited)** — vs 17–24 h per arm on
the transformers fallback stack.

## Final numbers

| Arm | Accuracy | Answered | Cap-hit | Mean tokens |
|---|---|---|---|---|
| Stock `Qwen/Qwen3.5-0.8B` | **8.4%** (42/500) | 9.8% | 90.6% | 30,108 |
| Original's published edit (layers 15/19 down_proj) | **48.6%** (243/500) | 83.2% | 26.4% | 20,645 |

**Corrected 2026-11-24 (ISSUES #19).** Both accuracies were re-scored with the final answer
checker (the checker was corrected twice that day). As first reported they were stock 6.0%
(30/500) and published edit 39.4% (197/500); after the first correction they read 7.0% (35/500)
and 43.8% (219/500). The Answered, Cap-hit and Mean-tokens columns are unchanged.

**Pre-registered verdict (phase3b thresholds): TRANSFERS.** The published edit scores **40.2 points
more than stock** (95% interval 35.8 to 44.6; 204 problems right only under the published edit,
3 only under stock) — far beyond the +10.0 bar. The share of generations that finish — ending
on their own instead of hitting the 32,768-token cap — flips from 9.4% to 73.6%. This is the
original's core claim, reproduced locally by us, on the user's own GPU.
(Corrected 2026-11-24, ISSUES #19: this paragraph said "+33.4 points, 6.6× relative" and
"Termination flips from 9% to 83% of generations"; 83.2% is the share that produced an answer, not
the share that finished.)

Anchor comparison (corrected 2026-11-24, ISSUES #19): stock 8.4 against openwsp's 7.2, Toledo's
independent 7.8% (93.1% cap-hit) and the original authors' 7.9% on their own stack — agrees.
Published edit 48.6 against openwsp's 50.6, Khaliq's 51.0 and the original's 48.1: within about
two and a half points of the outside replications (2.0 points below openwsp, 2.4 below Khaliq,
0.5 above the original authors' own number). Known differences, none yet tested: our benchmark
question is the bare problem while theirs adds "Please reason step by step, and put your final
answer within `\boxed{}`"; we take one sample per problem, they four; and the two graders do not
read every answer the same way — they agree on 82% of the published edit's rows (48.6 against
44.2 on the same generations): math-verify, given a whole answer section, takes the last `$…$`
span and so misses an answer such as "The value of `$x$` is 83" (ISSUES #19). A calibration run
on their exact protocol is pending. The earlier text put the shortfall at "~9 points shy" and
called the gap robust because "extraction micro-choices moved each arm by ≤1.6 points"; that was
wrong — the re-scoring alone moved the gap from +33.4 to +40.2.

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
   corrected twice on 2026-11-24 and all saved runs re-scored. (It replaces the sentence
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
   "10". What the final checker does (identically for every arm; it never looks at the gold
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
   refined 5,275; final checker 5,795.
   - Final checker against the rule as first committed: **1,125 answers turn from wrong to
     right and 11 from right to wrong.** Four of the 11 were accidental hits of the old rule (it
     took a digit out of text that was not an answer). Seven are misreads that remain in the
     final checker: a sentence inside display math; a value offered in two equal forms, the
     second taken (two answers); "There are `$76^\circ$` degrees in `$\angle B$`."; "There are
     7 such positive integers `$n$`."; "There are 4 possible values for `$x$`."; one looping,
     cut-off text. The checker is frozen here and these seven are on record (ISSUES #19): each
     is one answer of a 500-problem run, 0.2 points.
   - Final checker against the first re-score: **570 right, 1 wrong.** The one is the "4.76"
     false positive that rule let through (a model wrote "4.76" after "`$\frac{9}{2}$`
     expressed as a decimal is"; the checker took the fraction and scored it right against a
     gold of 4.5). The independent grader (math-verify), given only the last line of each of
     the 570 answers, agrees with 566.
   - Final checker against the first correction as last refined: **520 right, none wrong.**

   45 unit tests (`phase3c/test_rescore.py`) pass. Every saved run was re-scored with the final
   checker, and every re-scored file records the checker version that produced it. **Reproduce**
   (no GPU, no gunzip needed): `uv run python phase3c/rescore.py <run> [<run> ...]` reads the
   gzipped generations directly; `<run>` is the file stem, e.g. `uv run python
   phase3c/rescore.py p3c_stock p3c_c3_original`.

## What this closes

- The user's challenge ("why not the same or better results?") is answered as far as the pipeline
  goes: a reproducible local run — **wrong engine → right engine; wrong scorer → right scorer;
  and the original's claim lands: 8.4 → 48.6%** (re-scored 2026-11-24; 6.0 → 39.4% as first
  reported). The published edit now reads within about two and a half points of the outside
  replications of the same edit. Still open, and for the calibration run above to settle (it is
  pending): the last two and a half points or so. (Corrected 2026-11-24, ISSUES #19; this
  bullet said the challenge was "now fully answered", and after the first checker correction it
  put the remaining shortfall at "the last 4 to 7 points".)
- Weight-space editing stopping thinking-loops is confirmed **on third-party-independent
  hardware** (this repo) in addition to openwsp/Khaliq/Noble.
- Still unclaimed when this phase closed: the no-hint full-budget run. It has been run since
  (phase 10) — the first run we know of that offered a blind editor 50 passes, and no claim about
  the whole field. (Corrected 2026-11-24, ISSUES #23: this bullet said the run was "still unclaimed anywhere,
  including here" and called the Phase-3a blind NULL "one of the only no-hint data points in
  existence".)
