# Phase 1 target selection (2026-11-03)

Selection per PROTOCOL.md: greedy policy, 64 episodes, seed range 850000+.
Script: `phase1/pick_target.py` (renamed from `select.py` — see ISSUES.md #4).

| seed | greedy mean | std | min | max |
|-----:|-----------:|----:|----:|----:|
| 0 | **9.4** | 0.8 | 8 | 11 |
| 1 | 440.4 | 46.2 | 363 | 500 |
| 2 | 497.1 | 22.8 | 316 | 500 |
| 3 | 316.9 | 85.6 | 211 | 500 |
| 4 | 9.4 | 0.8 | 8 | 11 |
| 5 | 170.0 | 118.6 | 61 | 500 |
| 6 | 103.3 | 3.8 | 97 | 112 |
| 7 | 500.0 | 0.0 | 500 | 500 |
| 8 | 500.0 | 0.0 | 500 | 500 |
| 9 | 354.5 | 82.0 | 230 | 500 |
| 10 | 480.0 | 90.3 | 70 | 500 |
| 11 | 492.8 | 11.0 | 459 | 500 |
| 12 | 9.4 | 0.8 | 8 | 11 |
| 13 | 124.3 | 7.5 | 109 | 139 |
| 14 | 45.4 | 8.7 | 32 | 70 |
| 15 | 118.7 | 4.9 | 107 | 130 |
| 16 | 500.0 | 0.0 | 500 | 500 |
| 17 | 114.5 | 5.3 | 105 | 126 |
| 18 | 102.8 | 8.8 | 87 | 129 |
| 19 | 500.0 | 0.0 | 500 | 500 |

**TARGET = seed 0** (greedy mean 9.4; three-way tie with seeds 4 and 12, broken by seed order).
Copied to `phase1/target.pt`. Training curves were never saved (blinding, PROTOCOL.md).
