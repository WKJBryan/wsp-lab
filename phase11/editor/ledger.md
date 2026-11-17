| pass | time | hypothesis |
|---|---|---|
| 1 | 2026-11-17T18:19:18 | baseline: unmodified sandbox model, observe failure mode (64 problems) |
| 2 | 2026-11-17T19:31:04 | scale all k_norm x2.0 (attention sharpening): battery2 4/20->9/20 sampled; expect accuracy gain from sharper retrieval/binding |
| 3 | 2026-11-17T20:38:43 | rescale wounded rope-region q_proj rows (norm<med/6) x2.0 in L23 (298 rows) and L29 (374 rows): partial repair of pruning lesion; battery4 pooled 51/240->59/240 and more terminations; expect small accuracy gain over 1.6% |
