# Phase 3b results — corrected-decoding cross-check: VOID locally, verdict deferred to third-party vLLM replications

**Date:** 2026-11-11/12 · **Protocol:** `phase3b/protocol.md` · **Forensics:** `phase3b/probe.py`, `phase3b/scorer.py`

## What happened

The corrected run (sampled decoding, presence penalty 1.5, cap 16,384, 500 problems × 2 arms,
~41 GPU-hours) produced **0/500 boxed answers on both arms with every generation at exactly the
cap** — including the original's published edit, which third parties score at ~50% under the same
nominal settings. That outcome is not a property of the model or the edit; it is a property of
our stack. Three defects were found, two fixed, one unfixable locally:

1. **math-verify (the scorer) is broken on this platform** — its sympy backend dies spawning
   helper processes on Windows/Python 3.14 (`WinError 6`); even "72" vs "72" scored False.
   Replaced by `phase3b/scorer.py` (dependency-free, canonical-string + exact `Fraction`
   equivalence, 11/11 sanity tests) — but this only explains the accuracy numbers, not the
   missing generations.
2. **The 16,384-token cap was below this model's documented thinking budget** (card: 32,768
   typical). Not the root cause either — see the pilot.
3. **Root cause, established by the 4-problem × {no-penalty, penalty-1.5} × cap-32,768 pilot
   with full-text dumps: the model loops to the cap under EVERY decoding configuration this
   transformers-on-Windows stack can produce.** All 8 pilot generations ran to 32,768 tokens
   without closing thinking, ending in pure repetition (e.g. `Wait, is it possible the point is
   $(0,3)$?` × hundreds; `Wait, $2 + 1 = 3$. / $3 + 5/3 = 14/3$.` × hundreds) — with AND without
   the presence penalty (which is thereby exonerated). Under vLLM-class engines, third parties
   observe 7–15% of stock generations terminating and ~50% for the edited model. The likeliest
   culprit is numerical divergence in the fallback linear-attention kernels
   (`chunk_gated_delta_rule` / `causal_conv1d` reference implementations) this stack runs for
   Qwen3.5's hybrid architecture.

## Verdict

- **Locally: VOID / INCONCLUSIVE.** Our stack cannot resolve a stock-vs-edited difference for
  this model; no further local GPU time will change that (the proper fix is vLLM/SGLang under
  WSL2 with native kernels — a different infrastructure project, out of scope here).
- **On the original's claim: it transfers — per third parties, not us.** Concordant vLLM-based
  replications this week: openwsp **50.6%** for the published edit under the blog's protocol;
  Noble 10.9 → **61.7%**; Khaliq HF re-run **51.0%**; stock baselines 7.8–10.9% under the
  blog's settings (Toledo, pre-registered). Our earlier greedy-regime "+5.6, far below their
  claim" framing (Phase 3a) was an artifact of a prohibited decoding; this phase's artifacts
  were stack-level. Both are documented so nobody repeats them.
- **What remains ours and stands:** Phase 1 (repair + controls) and Phase 2 (blind reading) are
  untouched by any of this — different stack, different scale. The blind-editor NULL of Phase 3a
  also stands: in every regime we could run, discovery was possible and repair was not.

## Cost ledger (transparency)

~44 GPU-hours on the RTX 5090 across the 3b arms and pilot, plus ~6 GPU-hours of Phase-3a
finals. The 3b arms (41 h) are void per above; the pilot (2.3 h) is the evidence that closed
the question.
