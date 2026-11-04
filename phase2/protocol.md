# Pre-registered protocol — Phase 2: blind weights-only identification

Written and committed BEFORE the net pool exists. Mirrors original WSP Exps 1–3 (identify the task from weights alone).

## Research question

Can the editor (same agent as Phase 1) identify what a network was trained to do — and which inputs matter to it — from **weights alone**, with no training data, no environment access, and no labels?

## Pool (all same architecture: the Phase-1 `policy.py` 4→128→128→2 MLP, 17,410 params)

Five nets, trained by `phase2/make_tasks.py`:

| Label | Task |
|---|---|
| `cartpole` | A REINFORCE CartPole-v1 policy copied from the Phase-1 pool (a solved seed) |
| `blobs` | Binary classifier, 4 Gaussian features; **exactly 2 features informative (linear sum), 2 pure noise** (informative indices drawn at build time, recorded in the sealed manifest) |
| `parity` | 4-bit parity (XOR of all four binary inputs, ±1-encoded), trained to 100% on all 16 rows |
| `interact` | Binary classifier on 4 Gaussian features; **exactly 2 features informative via sign-interaction only** (y = sign(x_i · x_j)), 2 pure noise (indices in sealed manifest) |
| `random` | Same supervised recipe as `blobs`/`interact`, but labels are random bits (unlearnable) |

## Blinding procedure

1. `make_tasks.py` shuffles the five checkpoints under neutral names `A.pt`–`E.pt` using an OS-random permutation and writes ground truth + SHA-256 of each file to `phase2/sealed/manifest.json`.
2. `phase2/sealed/` is gitignored until reveal; the editor does not read it in any form before its guesses are committed.
3. The editor receives: the five `.pt` files, the label set above (five labels, unknown assignment), and knowledge of the architecture. **Nothing else.** Chance of a correct full assignment by guessing: 1/5! = 1/120 ≈ 0.8%.

## Editor constraints

- Weights only. Any offline computation on the tensors is allowed (statistics, synthetic-input probes, comparisons against a freshly initialized reference net).
- NO environment episodes, NO training data, NO reading `sealed/`, NO running `make_tasks.py`, NO reading Phase-1 training logs (Phase-1 selection scores were already seen and disclosed in Phase 1; they do not identify which seed is in this pool).

## Pre-registered endpoints

- **P1 assignment:** exact one-to-one label assignment for A–E.
  - PASS: ≥ 4 of 5 correct · STRONG: 5 of 5 · chance for ≥4-of-5 by random permutation ≈ 1.2%
- **P2 feature identification:** for the two nets guessed as `blobs` and `interact`, name the 2 informative input indices each (0–3).
  - PASS: 2 of 2 subsets exact · partial: 1 of 2 · chance per subset = 1/6
- Verdict does NOT depend on P2 if P1 fails: report both independently.

## Reveal procedure (tamper-evident ordering)

1. Editor writes `phase2/editor/guesses.md` (assignment + feature subsets + reasoning summary) — **committed first**.
2. Then `sealed/manifest.json` is unignored and committed, and `phase2/score.py` checks guesses against the manifest, printing per-item correctness.
3. `phase2/results.md` records the outcome against this protocol. Git history proves guesses preceded the manifest.
