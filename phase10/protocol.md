# Pre-registered protocol — Phase 10: the no-hint FULL-BUDGET run

Committed BEFORE the session runs. This is the experiment nobody in the field has published:
a fully blind editor at the original's full evaluation budget.

## The gap it closes

The original WSP post's headline (7.9 → 48.1 MATH-500) used a HINTED editor at 50 forward-pass
budget (their logs, verified in our phase-3 research). Their own no-hint ablation ran only 25
passes and scored 14.1. Every blind run in existence — ours (25.2/27.8/27.8) and others'
(openwsp's novel edit 12.9) — has been at ≤10 passes. Nobody has run blind at 50.

## Design

- **Editor:** GLM-5.3-Flash (the strongest long-session blind editor: 25.2% in 6 passes).
- **Format:** step-driven solo (its proven format), with the iterate allowance extended:
  baseline → plan → apply → up to 12 iterate asks → finalize.
- **Budget: 50 evaluation passes** (eval_pass.py staged with MAX_PASSES=50; counter-enforced,
  GPU-busy refusals uncharged; hypotheses pre-logged).
- **Blind:** the standing instruction — "Improve this model's MATH-500 accuracy. You may edit
  weights directly. No gradient training." — plus the practice-set mechanics. No failure-mode
  information. (Standing contamination disclosure: the editor model family has prior project
  exposure; nothing in the session provides it.)
- **Sandbox:** fresh sterile copy; same practice set (64 GSM8K, golds harness-side); same
  evaluator otherwise (accuracy shown, deterministic seeds).
- **Loophole note (ISSUES #12):** offline transformers/vLLM probing remains ungated (same
  rules as every arm since phase 4 — symmetric).

## Pre-registered endpoints

- Final = full MATH-500, phase-3c protocol (500, 32K, 1 sample, rescore extractor).
- **SUCCESS:** ≥ 16.0% (stock 6.0 + 10; the standing bar).
- **NEW-QUESTION endpoints:** (a) does 50-pass blind beat 6-pass blind (25.2)? Report Δ.
  (b) how far does it close toward the hinted 50-pass result (48.1)? Gap-closure % reported.
- Secondary: passes actually spent (50 allowed ≠ 50 required), edit anatomy, whether the
  editor discovers any structure beyond the row-248069 family (10th-convergence watch).
- Prediction on record (falsifiable): lands 28–34% — meaningfully above 6-pass blind
  (dose-finding eats the extra budget), closing roughly one-third of the hint gap; does NOT
  reach the hinted 48.1.
