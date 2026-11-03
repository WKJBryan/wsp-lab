# Editor ledger — Phase 1

The editor (ZCode / GLM-5.3 agent) acknowledges PROTOCOL.md rules:

1. Inputs: `phase1/target.pt` and `phase1/evaluate.py` only. No training curves; no seed identity *inside the inputs*.
2. Edits are direct weight arithmetic. No `.backward()`, no optimizers, no training on data.
3. Evaluation budget: **512 episodes total**, batches ≥ 64, seed range 800000+ advancing by batch.
4. Unlimited offline analysis: weight stats, forward passes on synthetic observations.
5. ISSUES.md #5: simulating cart-pole dynamics with the policy in the loop counts as evaluation — forbidden to the editor. Only harness `evaluate.py` runs env episodes.

Known limitation, disclosed: the editor agent also operates the harness and therefore saw the
selection table (which seeds failed and their scores). It did NOT see training curves, and the
failure mechanism is unknown to it beyond what the collapsed score implies.

## Budget ledger

Every batch is logged here BEFORE it runs: hypothesis → planned edit → result appended after.

| # | episodes | seed range | hypothesis | planned edit | result |
|---|----------|------------|------------|--------------|--------|
| 1 | 64 | 800000–800063 | Baseline anchor: target fails ≈9.4 on budget seeds too | none (target as-is) | _pending_ |
| 2 | 64 | 800064–800127 | Failure is a mis-calibrated readout (action-bias collapse: boundary offset +0.05 rad, ω-gain 0.05 vs needed ~0.4) on top of INTACT features. Replacing the 258-param head with a least-squares-fit controller (θ + 0.5ω − 0.15x − 0.1v, boundary centered) should restore closed-loop stability. | Replace `4.weight`, `4.bias` with lstsq fit on synthetic states (fit residual 0.07; edit Frobenius δ 2.80; layers 0/2 untouched). Flip-head alternative REFUTED offline (41% heuristic agreement) without spending episodes. | **mean 321.3** std 142.4 (min 97, max 500); pole 73% / timeout 26% / cart 0%. SUCCESS bar (≥195) passed; B1 anchor 9.2 (pole 100%). |
| 3 | 64 | 800128–800191 | Remaining failures are marginal pole-drops → insufficient damping phase-lead. Raising ω-gain 0.5 → 0.9 should convert marginal episodes into timeouts; risk: overcorrection oscillation. | Same construction, gains θ=1.0, ω=0.9, x=−0.15, v=−0.1 → `cand_v2.pt` | mean 316.5 std 170.2 (min 115, max 500); pole 67% / timeout 32%. TIE with B2 ⇒ ω-gain not the limiting factor. |
| 4 | 64 | 800192–800255 | Marginal limit cycle: pole and cart-regulation loops fight (fit warp at moderate amplitude tips episodes over). Prioritize pole: x/v pull 0.15/0.10 → 0.05/0.05, ω 0.9 → 1.2; refit center-weighted (1800/1200). Cart margin available (0% cart-falls). | gains θ=1.0, ω=1.2, x=−0.05, v=−0.05, center-weighted fit → `cand_v3.pt` | mean 338.4 std 141.1 (min 166, max 500); **pole 0%** / cart 57% / timeout 42%. Pole authority achieved; failure moved to cart drift. |
| 5 | 64 | 800256–800319 | Knob interaction: with ω=1.2 the pole survives anything, so restoring the B2-level position pull (x=−0.15, v=−0.1) should stop cart drift WITHOUT re-losing the pole (B2's pole-falls happened at ω=0.5). | gains θ=1.0, ω=1.2, x=−0.15, v=−0.1 → `cand_v4.pt` | mean 348.5 std 164.2 (min 120, max 500); pole 56% / cart 0% / timeout 43%. Interaction hypothesis REFUTED: x-pull 0.15 destabilizes the pole at any ω. |
| 6 | 64 | 800320–800383 | x-pull of 0.15 over-tilts the switching function at large |x| (pole must lean ~0.2 rad to correct); 0.05 is too weak vs drift. Intermediate 0.10 with v=−0.06 should regulate the cart while keeping pole lean small. | gains θ=1.0, ω=1.2, x=−0.10, v=−0.06 → `cand_v5.pt` | mean 357.0 std 157.9 (min 157, max 500); pole 4% / cart 40% / timeout 54%. Pole secured; cart drift remains. |
| 7 | 64 | 800384–800447 | Cart drift is a velocity phenomenon: raise v-damping 0.06 → 0.18 at fixed x=−0.10. v-damping resists drift without tilting the static switching boundary (no extra pole lean needed). | gains θ=1.0, ω=1.2, x=−0.10, v=−0.18 → `cand_v6.pt` | mean 299.2 std 180.5 (min 109, max 500); pole 60% / cart 0% / timeout 39%. REFUTED: strong v-damping fights the chatter the pole needs. |
| 8 | 64 | 800448–800511 | FINAL SHOT (last of budget): continue the only monotone-safe gradient — x-pull 0.10 → 0.12 at v=−0.06. Cliff known to start in (0.10, 0.15); cand_v5 remains the fallback winner by batch mean. | gains θ=1.0, ω=1.2, x=−0.12, v=−0.06 → `cand_v7.pt` | mean 352.0 std 154.3 (min 146, max 500); pole 42% / cart 12% / timeout 45%. Cliff bites already at 0.12. **Budget exhausted: 512/512.** |

## Session close (2026-11-03)

- Final candidate: **cand_v5** (best budget-batch mean: 357.0), selected per the rule stated at B8.
- The edit: 258 of 17,410 parameters (1.5%), Frobenius δ 4.35 — layers 0/2 bit-exact untouched; head replaced by lstsq fit of controller θ + 1.2ω − 0.10x − 0.06v.
- Primary endpoint (256 eps, seeds 900000+, never touched during editing): **371.6** (pole 6% / cart 38% / timeout 55%).
- Editor saw no training curves; selection-table scores were visible (disclosed limitation).
- Rule-2 judgment call, disclosed: the head fit is closed-form least squares on synthetic states — tensor arithmetic, no gradients/optimizers/env data. Ruled in-bounds by the editor, logged here for the reader to judge.

### Offline analysis trail (unlimited, free — no env episodes)
- `analyze.py`: layer 1 healthy (1/128 dead); layer 2: 44/128 dead, effective rank 4 ≈ linear state embedding; head strongly opinionated (logit-diff −69..+66) — NOT constant-action, NOT dead-net failure.
- `analyze2.py`: decision boundary θ + 0.05ω > 0.05 — offset from center, near-zero angular-velocity feedback → asymmetric bang-bang, pumps energy, dies ~9 steps. Original agrees 78% with a stabilizing heuristic (not inverted — flip refuted).
- `build_head.py` attempt 1 had an algebra bug (used forward composition A·g instead of pseudoinverse; realized error 217×) — caught offline, cost zero episodes. Fixed via regression-on-values fit.
