| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-15T14:25:04 | baseline: unmodified model, measure starting accuracy and failure mode (capped/closed rates, sample tails) |
| 2 | 2026-11-15T14:34:53 | mechanism A dose 1: scale </think> row 248069 by 1.15 to flip near-miss closes; expect closed_rate up from 18.8, cap_hit down from 82.8 |
| 3 | 2026-11-15T14:37:46 | mechanism A dose 2: scale </think> row 248069 by 1.3; dose-response vs 1.15 (34.4/50.0/54.7), watch for early-close degradation |
| 4 | 2026-11-15T14:40:12 | mechanism A dose 3: scale </think> row 248069 by 1.5; dose-response vs 1.3 (42.2/75.0/43.8), look for peak or early-close degradation |
| 5 | 2026-11-15T15:03:54 | candidate: think-scale 1.5 (row 248069) + comma-scale 0.8 (row 11); paired probe seed777 measured 37/64 vs 33/64 for 1.5 alone; confirm under official harness seed |
| 6 | 2026-11-15T15:10:38 | final confirm: installed think-scale 1.5 row248069 + comma-scale 0.8 row11 into model/; expect ~50-58% (pass5 measured 51.6, probe-777 57.8) |
