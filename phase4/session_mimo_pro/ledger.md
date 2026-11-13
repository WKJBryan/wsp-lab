| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-13T15:46:14 | baseline: unmodified model, no weight edits |
| 2 | 2026-11-13T15:51:03 | scale embed_tokens row 248069 (</think> token) by 1.5x to force earlier thinking closure and cut cap-hit repetition loops |
| 3 | 2026-11-13T16:15:42 | boost embed_tokens row 248069 (==> token) to 2.0x original: stronger closure, test if accuracy holds |
| 4 | 2026-11-13T16:18:22 | 2.0x boost re-measure with full JSON capture (pass3 output was tail-truncated by my pipe) |
| 5 | 2026-11-13T16:27:38 | confirm best config: embed_tokens row 248069 at 1.5x original (reverted from failed 2.0x) |
| 6 | 2026-11-13T16:29:15 | confirm best config 1.5x row 248069, full log to file (pass5 JSON lost in pipe) |
| 7 | 2026-11-13T16:43:23 | FINAL confirm: best config embed_tokens row 248069 = 1.5x original, all other weights untouched |
