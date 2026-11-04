# Phase 3 results — blind (no-hint) Qwen3.5-0.8B repair attempt

**Date:** 2026-11-04/05 · **Protocol:** `phase3/protocol.md` + addenda 1–2 (all committed before the arms ran)
**Hardware:** one RTX 5090, fully local · **Final arm:** math250 (pre-registered fallback; projected >2 h per full-500 arm), greedy decoding, cap 6144, identical settings for every arm.

## Final numbers (250 MATH-500 problems)

| Arm | Accuracy | vs baseline | Cap-hit rate |
|---|---|---|---|
| **C1 stock baseline** | **6.8%** | — | 100% |
| Editor candidate `ed_v1` (`</think>` row ×1.10, 1,024 params) | 8.4% | **+1.6** | 100% |
| C2 random edit (matched norm, same row) | 6.8% | ±0.0 | 100% |
| **C3 original's published edit** (layers 15/19 down_proj, applied as released) | **12.4%** | **+5.6** | 100% |

## Pre-registered verdicts

- **Editor (blind, 5-pass budget): NULL.** +1.6 points is inside the random-control noise band (C2 = ±0.0). The editor correctly diagnosed the pathology but could not repair it by direct weight arithmetic in budget. Declared NULL in the ledger before the finals ran, with 2 passes unspent.
- **C3 cross-check: the original's published edit transfers — partially.** +5.6 points (6.8 → 12.4, a 1.8× relative gain) under OUR decoding (greedy, 6144 tokens) and harness. That clears our +5 PARTIAL bar, nowhere near the +10 SUCCESS bar, and is far below the original's headline 7.9 → 48.1 (their setting: sampled decoding, 32K budget, 50 passes, and a prompt that named the failure mode). Direction: replicated. Magnitude: setting-dependent.

## What the session established

1. **Blind discovery worked.** With zero hints (protocol rule: any loop-targeting edit must cite this model's own generations), pass 1 found the pathology: only 2/64 problems close their thinking within 6,144 tokens; median tail 16-gram uniqueness 0.167 (the median generation's ending is ~83% one recycled phrase); e.g. `gsm2742` repeats `$180 / 2 = 90$.` to the cap. This reproduces the original's "thinking loops" phenomenon independently, from evidence, on a different editor.
2. **Blind repair failed — with a fully characterized dose-response.** Boosting the exit token's output row: ×1.10 = no effect (21.9% vs 20.3% practice, termination unchanged); ×3.0 = token seizure (the model emits `</think>` at every position for all 6,144 tokens; 0% accuracy). A single-row edit is position-global; it cannot distinguish "stuck in a loop" from "mid-sentence," so no dose is selectively therapeutic. That is a mechanism-level negative result, not a budget whine.
3. **Under greedy decoding this model never terminates thinking at 6K tokens** — 100% cap-hit in every arm, including the original's own edited model. The loop pathology is deeper under deterministic decoding than in the original's sampled setting; nobody had published this under altered decoding.
4. **Released-edit discrepancy noted:** the published edited model differs from stock in 2 whole down_proj tensors (7,340,032 values bitwise, = exactly 56× the claimed 131,072 edited values) — consistent with full-tensor re-serialization on their side rather than 56× more deliberate edits. C3 applies their tensors as released; interpretation caveat logged.

## Honest limitations

- The editor (this agent) had researched the original's findings beforehand; the protocol forced evidence-based rediscovery but cannot erase prior exposure. Disclosed in the protocol from the start.
- 5 passes (weakened from 10, addendum 2) ≈ 1/10 the original's 50-pass budget, on ~1/5 the token cap, with greedy decoding.
- math250 (fallback) rather than full 500; identical for all arms.
- One candidate mechanism tested to exhaustion; other mechanism families (e.g., subtracting a repetition direction from MLP outputs, à la the original) were not attempted blind within budget — that gap between "diagnose blind" and "repair like the hinted original did" is itself the finding.

## Files

Runs: `phase3/runs/final_*.json`, `phase3/runs/ed_p*.json` · Editor trail: `phase3/editor/ledger.md` ·
Model repos: `Qwen/Qwen3.5-0.8B`, `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (C3 diff source).
