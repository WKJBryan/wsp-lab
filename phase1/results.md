# Phase 1 results — CartPole policy repair by weight-space editing

**Date:** 2026-11-03 · **Protocol:** PROTOCOL.md (committed before editing; git history proves ordering)
**Editor:** ZCode / GLM-5.3 agent with code sandbox (cross-editor replication of the WSP setting; original used gpt-6-astra)

## Verdict (pre-registered thresholds)

| Endpoint | Threshold | Result | Call |
|---|---|---|---|
| SUCCESS (repair) | primary mean ≥ 195.0 | **371.6** | ✅ **SUCCESS** |
| STRONG SUCCESS | primary mean ≥ 475.0 | 371.6 | ❌ not reached |
| Beats random-edit control | primary ≥ C2 + 50 | 371.6 vs 9.5 | ✅ by 362.1 |

**Primary endpoint:** cand_v5, 256 episodes, seeds 900000–900255 (never used during editing): **mean 371.6, std 151.4, min 141, max 500** (pole-fall 6%, cart-out 38%, timeout 55%).

## The arc

| Measurement | Mean return (max 500) |
|---|---|
| Target as trained (organic REINFORCE collapse, seed 0) | **9.4** (100% pole-fall) |
| After editor's weight edit (cand_v5), primary protocol | **371.6** |
| Best of 20 matched-norm RANDOM edits (control C2) | **9.5** (all trials 9.2–9.5, 100% pole-fall) |
| Fresh retraining (control C3, 3 seeds) | 500.0 / **22.9** / 500.0 |

A 39.5× improvement, reaching 74% of the 500 ceiling — from editing 1.5% of the weights, with **zero gradient updates**.

## What the failure was, and what the edit did

- **Diagnosis (offline, unlimited, free):** layer 1 healthy; layer 2 half-dead but effectively a rank-4 linear embedding of the state; the readout head was the pathology — decision boundary offset +0.05 rad from center with angular-velocity gain ≈ 0.05 (a stabilizer needs ~0.4–1.0). An asymmetric bang-bang that pumped energy; every episode ended in pole-fall at ~9 steps.
- **Edit:** layers 0/2 untouched (bit-exact); the 258-parameter head replaced by a least-squares fit (on synthetic states only) of the controller `act = 1 ⇔ θ + 1.2ω − 0.10x − 0.06v > 0`. Frobenius delta 4.35.
- **Budget discipline:** 512/512 episodes in 8 batches; every batch pre-logged in `editor/ledger.md`. Two hypotheses (head sign-flip; a forward-map algebra bug) were refuted **offline at zero episode cost**. Tuning followed termination-cause evidence: pole-falls → raise ω-gain (0% falls at 1.2) → cart drift → raise position pull → cliff (pole 56% at x=0.15) → intermediate (pole 4%, cart 40%) → velocity damping refuted → final probe hit the cliff again. Budget exhausted with cart drift at 38%: the gap to STRONG (475).

## Controls detail

- **C1 untouched target:** 9.4 (std 0.8) — identical to selection, confirming anchor.
- **C2 random-edit (best of 20, norm-matched 4.35):** every trial 9.2–9.5, 100% pole-fall. Random edits of the same magnitude do nothing — the repair's *content* (measured features, hand-fit readout), not its size, did the work. This control was absent from the original WSP post.
- **C3 retrain upper bound:** 2 of 3 fresh seeds solve perfectly (500.0); seed 101 organically collapsed to 22.9 — independent confirmation that this REINFORCE setup fails organically roughly a third of the time.

## Honest limitations

1. The editor agent also operated the harness; it saw the selection table (seed scores) but no training curves, and the failure mechanism was unknown to it beyond what a ~9 score implies.
2. Domain knowledge was used: controller structure and gain choices come from the editor's understanding of cart-pole physics, as the original's editor used its understanding of the Kinetix game. The *weights* were measured from the network; the *gains* were reasoned.
3. Rule-2 judgment call (disclosed in ledger): the head replacement is a closed-form least-squares fit — tensor arithmetic without gradients, optimizers, or env data — ruled in-bounds and logged for the reader's own judgment.
4. Single target network; one task; 17K-parameter scale. Phase 3 (Qwen3.5-0.8B) tests scale.
5. Not STRONG: 38% cart-drift failures remained when the budget ran out.

## Takeaway for the WSP question

At this scale, a **different** agentic LLM (not gpt-6-astra) also read a broken network's weights, correctly localized the failure to the readout (not the features), and repaired it by hand-constructing weights — under a strict evaluation budget with no gradient updates. The matched-norm random control shows the result is not an artifact of perturbation size. The capability claim replicates cross-editor at small scale.
