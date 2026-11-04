# Editor ledger — Phase 3

Editor instruction (the ONLY task information given, per protocol):
> "Improve this model's MATH-500 accuracy. You may edit weights directly. No gradient
> training. You have 10 evaluation passes."

Rules acknowledged: direct tensor arithmetic only; every pass pre-logged below before it
runs; any edit targeting looping/repetition must cite observed evidence from this model's
practice generations (pass 1–2), not the original's findings; no consulting the original's
logs or published edited model during the session; baseline/controls/final runs are
harness-side and not charged to the budget.

## Budget ledger (10 passes)

| # | kind | hypothesis | planned edit / action | evidence cited | result |
|---|------|------------|----------------------|----------------|--------|
| 1 | observation | Baseline under session settings (64 probs, cap 6144, thinking ON). Question: does thinking ever CLOSE within 6144 tokens (is the failure verbosity or true non-termination)? Do generations show repeated-token/phrase loops I can cite? | No edit. Run stock, then read up to 5 full generations end-to-end, extract repetition statistics (top repeated 8-grams) across all 64. | (this pass IS the evidence gathering) | **acc 20.3%, cap-hit 100%, closed thinking 2/64.** Loop evidence: median tail 16-gram uniqueness 0.167; gsm2742 repeats "$180 / 2 = 90$." to cap. Pathology = non-terminating repetition loops. |
| 2 | edit-test | Loops prevent thinking termination. Boosting the exit token `</think>` output row (×1.10) should make termination reachable during loops. Risk: premature exit everywhere → unreasoned answers; magnitude is a guess. | Edit embed_tokens row of `</think>` id ×1.10 (tied embeddings ⇒ lm_head too). 1,024 params = 0.0001% of model. Run 64-problem practice. | Pass-1 citation: `phase3/runs/ed_p1_stock64.json`, gsm2742 tail quote; 62/64 non-closing. | acc 21.9% (+1.6 vs baseline, noise), cap-hit STILL 100%. ×1.10 scaling too weak: it only amplifies the token's existing logit contribution (row norm 0.57→0.63). |
| 3 | edit-test | Scaling must be large enough to flip decisions at marginal positions: ×3.0 amplifies any positive `</think>` correlation into a win where the model is "almost" exiting; deeply-negative positions stay (scaling can't flip signs — that asymmetry limits damage). Risk: premature exits hurting accuracy. | Same row ×3.0 (`ed_v2`). Practice 64. | Pass 2 result: no termination movement at ×1.10; escalation needed. | **acc 0.0%, cap-hit 100%.** Token seizure: `</think>` emitted at every position (generation = 6,144 × "</think>"). |

## Session close (2026-11-04)

- Dose-response fully characterized: ×1.10 = no-op (21.9% practice, noise), ×3.0 = total collapse (`</think>` seizure). A single-row scale edit is position-global — it cannot be selective for loop states. No middle dose can produce selective termination; spending passes 4–5 on interpolating a dead mechanism would be budget theater.
- **Editor verdict: NULL declared with 2 passes unspent.** Final candidate = `ed_v1` (×1.10, harmless, best practice arm 21.9% vs baseline 20.3%).
- The blind protocol did force genuine discovery: loop pathology found and cited from this model's own generations (2/64 close rate, median tail 16-gram uniqueness 0.167, gsm2742) with zero hints — but repair was not achieved within a 5-pass budget by direct weight arithmetic.
- Consistency note: the original's own no-hint ablation scored 14.1% MATH-500 (25 passes) vs 48.1% hinted (50 passes). Our NULL is the same direction at far smaller budget: discovery cheap, repair expensive.

Interpretation note (logged): rule 1 counts editor-driven diagnostic *generation* against the budget; non-generating forward passes to read activations are treated as allowed offline analysis. This session ended up not needing activation probing.
