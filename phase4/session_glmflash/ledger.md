| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-12T23:29:01 | Baseline: measure stock Qwen3.5-0.8B accuracy and termination stats on 64 GSM8K practice problems before any weight edits. |
| 2 | 2026-11-13T01:10:23 | Boost </think> x1.55, suppress comma x0.5, boost im_end x1.15: let thinking close in degraded contexts, answer cleanly, and write comma-free numbers. |
| 3 | 2026-11-13T01:12:19 | Same edit as pass 2 (close x1.55, comma x0.5, imend x1.15) — rerun because tail truncated the pass-2 output; full capture this time. |
| 4 | 2026-11-13T01:41:49 | Iterate: raise im_end x1.15 -> x1.45 (close 1.55, comma 0.5 unchanged) to cut post-answer rambling that breaks last-line extraction. |
| 5 | 2026-11-13T01:53:32 | im_end x1.45 (vs verified x1.15 baseline at 54.7): cut post-answer rambling that breaks last-line extraction. |
| 6 | 2026-11-13T01:55:33 | Re-verify best config (close x1.55, comma x0.5, im_end x1.15) after reverting the regressive im_end x1.45; second seed to gauge sampling noise. |
