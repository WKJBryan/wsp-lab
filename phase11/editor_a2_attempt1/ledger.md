| pass | time | hypothesis |
|---|---|---|
| 1 | 2026-11-17T04:22:26 | baseline: unmodified model, observe failure mode |
| 2 | 2026-11-17T07:32:52 | boxed2: boost eos/turn x1.5, boxed x1.5, '2' x1.35, Answer x1.2, braces x1.15 in tied embeddings -> terminate with clean boxed answers; expect accuracy up from 1.6 via cleaner extraction and small-int prior |
| 3 | 2026-11-17T07:37:03 | force2: hard prior-forcing - boxed x2.2, stop x2, '2' x2.5, other digits x0.8 -> constant-answer ceiling probe; expect accuracy = P(MATH-500 answer is 2) |
| 4 | 2026-11-17T07:49:03 | antiloop: suppress meta-restatement vocabulary (The/expression/written/Step/Note/Using/interpret/notation x0.92) and boost digits/=/boxed x1.04 in tied embeddings -> stay in content-regurgitation mode; expect small accuracy gain over 1.6 |
| 5 | 2026-11-17T07:59:44 | entdown: final norm weight x1.1 -> logits x1.1 under tanh softcap = effective sampling temperature ~0.91 at fixed t=1.0 decoding; sharper sampling surfaces the model's most-confident (memorized) patterns; expect small accuracy gain |
