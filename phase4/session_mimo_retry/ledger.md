| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-13T11:28:42 | baseline evaluation of unmodified model |
| 2 | 2026-11-13T14:39:06 | scale embed_tokens row 248069 (end-of-thinking) by 1.5 to raise closure and accuracy |
| 3 | 2026-11-13T14:41:42 | EOT row factor 1.3x (revert to backup then rescale): probe for accuracy peak below 1.5x |
| 4 | 2026-11-13T14:44:37 | EOT row factor 1.7x: test upward of the 1.5x optimum for accuracy |
| 5 | 2026-11-13T14:47:16 | EOT row factor 1.6x: bracket the 1.5x peak between 1.5 and 1.7 |
| 6 | 2026-11-13T14:49:31 | confirm 1.5x EOT factor reproducibility under a new seed (pass 6, seed=SEED+6 vs pass2 SEED+2) |
| 7 | 2026-11-13T14:52:21 | final confirmation: best config (EOT row 248069 x1.5) held on disk before reporting |
