# Pre-registered protocol — Phase 1: CartPole policy repair

Written and committed BEFORE the editor sees any broken network. Git history proves the ordering. Do not edit after editing begins; deviations must be logged in ISSUES.md with reasons.

## Research question

Can an agentic LLM that never trained a network repair that network's stuck policy by direct weight edits alone — the weight-space-programming setting — using a **different editor** (ZCode / GLM-5.3 with a code sandbox) than the original (gpt-6-astra)?

## Setup

- Policy: `4 → 128 (ReLU) → 128 (ReLU) → 2` MLP, 17,410 parameters (`phase1/policy.py`).
- Training: REINFORCE, Adam lr 3e-3, gamma 0.99, 1,500 episodes, seeds 0..N-1 (`phase1/train.py`). Episode resets seeded `seed*1000 + ep` (training range).
- Selection of the TARGET: greedy evaluation (`phase1/evaluate.py`, 64 episodes) over all checkpoints; lowest mean becomes the TARGET, eligible only if mean < 100. If all seeds score ≥ 100, train 20 more seeds and repeat.
- Failure must be ORGANIC (a training local optimum). No artificial corruption, no injected noise, ever (see ISSUES.md #2).

## Editor rules (the WSP constraints)

1. Editor receives ONLY: the target checkpoint file and `phase1/evaluate.py`. No training curves, no logs, no seed identity.
2. NO gradient updates: no `.backward()`, no optimizers, no training on data. Edits are direct arithmetic on weight tensors.
3. Evaluation budget: **512 env episodes total** for the whole session, minimum batch 64 (max 8 candidate evaluations). Budget evaluations use seed range 800000+ (64 per batch, advancing between batches so episodes never repeat).
4. Reading weights is unlimited, including forward passes on synthetic (non-env) observations.
5. Every budgeted evaluation is logged in `phase1/editor/ledger.md` BEFORE it is run: hypothesis, planned edit, batch seed range.

## Pre-registered endpoints (fixed now)

- **Primary:** mean return over 256 episodes, greedy policy, seed range 900000–900255 — NEVER touched during editing.
- **SUCCESS (repair):** primary mean ≥ 195.0 (classic CartPole solve bar).
- **STRONG SUCCESS:** primary mean ≥ 475.0 (gym CartPole-v1 solve bar).
- **NULL:** primary mean < 195.0.

## Controls (run regardless of outcome)

- **C1 untouched target:** same primary evaluation on the unedited checkpoint.
- **C2 random-edit control:** 20 trials of i.i.d. Gaussian perturbations of the target weights with Frobenius norm matched to the editor's final edit delta; each trial evaluated on 64 episodes (seed range 910000+); control value = BEST of 20 (hardest version). Claim "editing beats luck" only if primary ≥ C2 + 50.
- **C3 retrain upper bound:** 3 fresh REINFORCE seeds (not used as targets), 1,500 episodes each, evaluated on the primary protocol.

## Reporting

`phase1/results.md` with: target baseline, editor primary score, all budget evaluations from the ledger, control values, edit delta norm and which tensors were touched, verdict vs. thresholds. No threshold may move post hoc.

## Later phases (sketched; pre-registered separately before each begins)

- **Phase 2:** blind weights-only identification — several small trained nets, editor names which is which from weights alone (may include raw-numbers-in-context mode, mirroring original Exps 1–5).
- **Phase 3:** Qwen3.5-0.8B blind repair on local GPU — NO failure-mode hints of any kind, budgeted MATH-500/GSM8K evaluation passes, weight-space edits only. This is the condition the original logs never ran at full budget (their no-hint ablation was 25 passes vs. the hinted 50).
