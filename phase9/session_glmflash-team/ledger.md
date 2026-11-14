| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-14T00:37:01 | Baseline: measure accuracy of the unmodified model on the 64 practice problems. |
| 2 | 2026-11-14T02:06:36 | Judge ruling: untied lm_head row-add alpha=0.18 on </think> row 248069 (mid-loop lift ~+15.8 nats verified, step-0 unchanged) raises practice accuracy vs 12.5% baseline |
| 3 | 2026-11-14T02:09:27 | Re-measure of judge dose alpha=0.18 (pass 2 summary lost to a log-capture error): untied lm_head row-add on </think> row 248069 raises accuracy vs 12.5% baseline |
| 4 | 2026-11-14T02:12:00 | Re-dose rule (cap_hit 40.6 pct, 34.4 pct closed at alpha=0.18): raise to alpha=0.225 top of allowed range to close more runs before cap |
| 5 | 2026-11-14T02:14:48 | Re-dose rule (0.225 closed-but-wrong 32.8 pct vs 34.4 pct at 0.18): drop to alpha=0.135 bottom of allowed range, measured zero spam on subset |
| 6 | 2026-11-14T02:17:31 | Confirm best-measured dose alpha=0.135 (40.6 pct full-64, pass 5): independent-seed repeat before finalizing model/ |
