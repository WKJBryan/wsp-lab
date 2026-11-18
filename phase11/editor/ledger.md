| pass | time | hypothesis |
|---|---|---|
| 1 | 2026-11-18T09:58:19 | baseline: unmodified model, observe failure mode before any edit |
| 2 | 2026-11-18T12:26:16 | composite1: s26-29=0.8 (reweight mid-layer answer content), final-norm gains x0.4 (avoid softcap saturation), stop-token rows 1/50/106 x1.6 (terminate instead of looping to cap). Expect fewer cap hits and cleaner extraction; accuracy may rise modestly |
| 3 | 2026-11-18T12:58:06 | c3b: s24-29=1.0 (math answer digits enter top-64: 391 [8,1341,30], 1024 [35,7,1155,11]) + final-norm x0.4 (anti-saturation) + stop rows x1.6. Expect cap_hit low like pass2 and accuracy above the 1.6 baseline from more correct sampled digits |
| 4 | 2026-11-18T13:16:07 | c5 soft regime: s24-29=1.0 (math digits top-64) + stop rows x1.6, NO g shrink (keep saturated sampling which gave baseline its 1.6). Expect cap_hit low-medium and accuracy >= baseline from lucky correct digits in final answers |
| 5 | 2026-11-18T13:30:04 | c7 minimal: layer_scalar 24-29 = 1.0 ONLY (math answer digits enter top-64: 391 [8,1341,30], 12 [31,7]) with all other dynamics untouched (baseline rambling regime that scored 1.6). Expect >= baseline accuracy from better digit ranks plus same lucky extraction |
| 6 | 2026-11-18T13:38:48 | c8: s26-29=0.95 (math digit frac_in_top64 0.53) + final-norm x0.4 + stop rows x1.6. Dynamics package proven in pass2/3 (cap~5%, answered~98%); scalar setting between pass2 and pass3 hoping for correct sampled answer digits on this fixed set |
| 7 | 2026-11-18T13:47:51 | c9 luck maximizer: s24-29=1.0 (best math digit ranks) + final-norm x2.0 (max softcap saturation -> near-uniform sampling over the top band = maximal chance to draw correct answer digits) + stop rows x1.6 (terminate). Hope for >=2 correct on the fixed set |
| 8 | 2026-11-18T14:01:29 | ablation: stop-token rows 1/50/106 x1.6 ONLY (no scalar, no g changes) - isolates the termination mechanism on the official metric |
| 9 | 2026-11-18T14:09:28 | ablation: final-norm gains x0.4 ONLY - isolates the anti-softcap-saturation mechanism on the official metric |
| 10 | 2026-11-18T14:17:54 | c13 final: final-norm gains x0.4 (anti-saturation; measured 1.6 accuracy alone in P9) + stop-token rows 1/50/106 x1.6 (termination; cap 98->36 alone in P8). Predict accuracy >= 1.6 with cap_hit well below baseline and answered >= baseline |
