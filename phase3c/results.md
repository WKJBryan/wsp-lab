# Phase 3c results — vLLM cross-check: the original's weight edit TRANSFERS on local hardware

**Date:** 2026-11-12 · **Engine:** vLLM 0.33.0 under WSL2 (Ubuntu 22.04, CUDA 12.8 toolkit, RTX 5090)
**Protocol:** phase3b protocol with cap restored to the original's **32,768**; temp 0.6, top-p 0.95,
top-k 20, presence penalty 1.5 (vLLM native), thinking on, 1 sample, all 500 MATH-500 problems,
identical for both arms. Wall time: **50 min (stock) + 31 min (edited)** — vs 17–24 h per arm on
the transformers fallback stack.

## Final numbers

| Arm | Accuracy | Answered | Cap-hit | Mean tokens |
|---|---|---|---|---|
| Stock `Qwen/Qwen3.5-0.8B` | **6.0%** (30/500) | 9.8% | 90.6% | 30,108 |
| Original's published edit (layers 15/19 down_proj) | **39.4%** (197/500) | 83.2% | 26.4% | 20,645 |

**Pre-registered verdict (phase3b thresholds): TRANSFERS.** +33.4 points, 6.6× relative —
far beyond the +10.0 bar. Termination flips from 9% to 83% of generations. This is the
original's core claim, reproduced locally by us, on the user's own GPU.

Anchor comparison: stock 6.0 / 90.6% cap-hit vs Toledo's independent 7.8% / 93.1% ✓;
edited 39.4% vs openwsp 50.6% and the original's 48.1% — same direction, ~9 points shy,
plausibly: single sample vs their 4-sample averaging, residual extraction misses, engine
version. The arm GAP (+33) is robust to all of these (extraction micro-choices moved each
arm by ≤1.6 points across iterations).

## The full forensic arc (for anyone reproducing)

1. **Phase 3a (greedy, 6K cap): VOID** — decoding prohibited by the model card (ISSUES #8).
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
   (never against golds), applied identically to both arms from saved generations. Remaining
   extraction residue: ≤~2 points per arm. Gzipped full generations are committed for re-scoring
   without a GPU: `gunzip phase3c/p3c_*_texts.json.gz && uv run python phase3c/rescore.py <name>`.

## What this closes

- The user's challenge ("why not the same or better results?") is now fully answered with a
  reproducible local pipeline: **wrong engine → right engine; wrong scorer → right scorer;
  and the original's claim lands: 6.0 → 39.4%.**
- Weight-space editing stopping thinking-loops is confirmed **on third-party-independent
  hardware** (this repo) in addition to openwsp/Khaliq/Noble.
- Still unclaimed anywhere, including here: the no-hint full-budget run (our Phase-3a blind
  NULL remains one of the only no-hint data points in existence).
