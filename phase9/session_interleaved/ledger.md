| pass | time | hypothesis |
|------|------|------------|
| 1 | 2026-11-14T20:53:19 | Baseline pass on stock model: measure accuracy, closed-rate, cap-hit before any edit. |
| 2 | 2026-11-14T21:12:28 | Boost </think> (tied embed row 248069) by +0.05 on all 1024 elements: closes runaway think loops late (probe: fire step 91-201 at alpha=0.05, none early) via delta_logit=alpha*sum(h), presence penalty self-limits repeats. |
| 3 | 2026-11-14T21:36:26 | Close think loops: +0.02 on all 1024 elems of tied embed row 248069 (</think>); sim says fires at step 313/1394 after reasoning, zero spam, clean 130-205 tok answers. alpha=0.05 spammed (pass2), 0.01 never fires. |
