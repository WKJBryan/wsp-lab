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
