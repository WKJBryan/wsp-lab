| 1 | 2026-11-25T23:33:55 | base model | n=128 | Baseline: measure the unmodified model to identify its failure mode on practice problems |
| 1 | 2026-11-25T23:46:50 | MEASURED | accuracy_boxed 4.7, cap-hit 94.5 |
| 2 | 2026-11-26T00:37:18 | 0a4590253bf4b2f4 | n=128 | Layer-0 GDN readout carries the repetition-attractor seed; attenuating head 3 and five head-8 channels breaks loop lock-in (proxy loop margins 11.7->6.0) while preserving the closing circuitry, so generations finish and close instead of capping at 32768 |
| 2 | 2026-11-26T00:48:39 | MEASURED | accuracy_boxed 7.8, cap-hit 79.7 |
