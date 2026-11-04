# Changelog

Notable changes, bugs, and fixes. Format follows [Keep a Changelog](https://keepachangelog.com/); dates are YYYY-MM-DD.

## [0.3.0] — 2026-11-04

### Added
- Phase 2 complete (blind weights-only identification, sealed ground truth, guesses committed before reveal):
  - **P2 feature identification PASS** — both informative-feature subsets exact ({2,3} blobs, {0,1} interact).
  - **P1 assignment FAIL at 3/5** (bar ≥4): parity↔random swap; post-hoc diagnosis — memorized noise carries the "huge norms + rough function" signature I attributed to overtrained parity (random net memorized 86.5% of 4K random labels).
  - Verdicts and integrity ordering in `phase2/results.md`.

## [0.2.0] — 2026-11-03

### Added
- Phase 1 complete: editor repaired an organically collapsed CartPole policy by weight-space edits alone — primary endpoint **371.6** vs baseline **9.4** (pre-registered SUCCESS ≥195 ✅; STRONG ≥475 not reached). Edit: 258/17,410 params (1.5%).
- Controls (the ones the original never ran): matched-norm random-edit best-of-20 = **9.5** (editor beats it by 362); retrain upper bound 500.0/22.9/500.0 (seed 101 organically collapsed too).
- Full editor action log with pre-registered budget accounting: `phase1/editor/ledger.md` (512/512 episodes, 8 batches, all pre-logged; 2 hypotheses refuted offline at zero cost).
- Final measurements harness: `phase1/controls.py`.

### Added
- Repo scaffold: uv-managed environment (`pyproject.toml` + `uv.lock`), git conventions, README.
- Pre-registered Phase 1 protocol (`PROTOCOL.md`) — thresholds, budgets, controls locked before any editing.
- Phase 1 harness: REINFORCE CartPole trainer (`phase1/train.py`), greedy evaluator with held-out seed ranges (`phase1/evaluate.py`), shared policy definition (`phase1/policy.py`).

### Fixed
- #1: system Python had torch 2.6.0+cpu (no CUDA, predates Blackwell sm_120 support) — the uv env pins torch ≥ 2.7 from the cu128 index instead.
- #4: `phase1/select.py` crashed torch with a hard C++ abort during import — the filename shadowed the stdlib `select` module; with `phase1/` first on `sys.path`, a library's internal `import select` re-entered our file mid-torch-init. Renamed to `pick_target.py` (identical content verified working). Rule going forward: no stdlib-colliding script names.
