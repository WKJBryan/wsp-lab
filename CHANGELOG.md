# Changelog

Notable changes, bugs, and fixes. Format follows [Keep a Changelog](https://keepachangelog.com/); dates are YYYY-MM-DD.

## [0.1.0] — 2026-11-03

### Added
- Repo scaffold: uv-managed environment (`pyproject.toml` + `uv.lock`), git conventions, README.
- Pre-registered Phase 1 protocol (`PROTOCOL.md`) — thresholds, budgets, controls locked before any editing.
- Phase 1 harness: REINFORCE CartPole trainer (`phase1/train.py`), greedy evaluator with held-out seed ranges (`phase1/evaluate.py`), shared policy definition (`phase1/policy.py`).

### Fixed
- #1: system Python had torch 2.6.0+cpu (no CUDA, predates Blackwell sm_120 support) — the uv env pins torch ≥ 2.7 from the cu128 index instead.
- #4: `phase1/select.py` crashed torch with a hard C++ abort during import — the filename shadowed the stdlib `select` module; with `phase1/` first on `sys.path`, a library's internal `import select` re-entered our file mid-torch-init. Renamed to `pick_target.py` (identical content verified working). Rule going forward: no stdlib-colliding script names.
