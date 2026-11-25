# Phase 14 weight-edit session notes

Budget: 15 passes. Pass 1 = baseline (running).

## Plan
1. Baseline pass, read generations, diagnose failure.
2. Teacher-forced reads (free) to localize mechanism.
3. Targeted weight edits, checker, measure.

## Findings

## Weight survey (CPU, while pass 1 ran)
- 320 LM tensors, bf16/f32; no NaNs; no dead heads/slices; layernorm near-zero channels random (normal).
- GDN decay g=-exp(A_log)*softplus(a+dt_bias); dt_bias bimodal per layer (fast/slow heads) - looks trained, not broken.
- L0 dt_bias rms 6.07 (others ~3), A_log extreme on 2 heads; L0 conv/conv taps normal.
- Model is Qwen3.5-0.8B post-trained (README). Full-attn layers have gated attention (q_proj = q+gate 2048+2048).

## Diagnosis (pass 1 baseline: acc 4.7%, closed 5.5%, cap-hit 94.5%, mean 31286 tok)
- Failure: model falls into VERBATIM repetition attractor ('Wait, ...' cycles), never emits </think>, caps at 32k.
- 7/128 gens closed; 6/7 of those CORRECT. Closing pathway works when reached.
- At loop entry entropy collapses ~0.3->0.0 (nats); loop margins 8-16 nats >> presence penalty 1.5.
- </think> reachable (logprob ~-2, sometimes top-1) ONLY at 'ready' contexts ('The final answer is X.\n', 'Okay. Ready.');
  mid-derivation P(</think>) ~ -14..-25 (healthy).
- Loop carriers: layer-0 GDN branch is THE seed (zero it: loop dies, top-1 ' Wait' -0.0 -> '=' -1.7);
  6 full-attn layers + GDN chain + MLPs form a redundant resonance (any one set zeroed breaks it, single layers don't).
- L0 head 8 (decay=1.0) = closing-support; h3 = loop-carrier; per-channel map found loop-only channels of h8 (44,50,76,81,120).
- Closing carried by L0-GDN + L23-MLP only (zero either: P(th) -2 -> -10..-13).

## Edit 1 (pass 2) = cand_C on pristine:
- layers.0.linear_attn.out_proj.weight: cols [384:512) (head 3) x0.5; cols 8*128+{44,50,76,81,120} zeroed.
- Proxy suite: loop mean 11.7 -> 6.0 (4/12 < 3), close -2.1, sanity OK ('4','408','capital').
- Earlier attempts: decay-target edit of dt_bias/A_log (model_try1): NO loop effect (loop is not state-decay-driven) -> dropped.
- h3 block x0 and x0.25 break simple arithmetic ('2+2=2') -> x0.5 is the functional edge.

## Pass 2 results (cand_C, fingerprint 0a4590253bf4b2f4)
- accuracy 4.7 -> 7.8 (10/128); closed 5.5 -> 26.6 (7 -> 34); cap-hit 94.5 -> 79.7; mean tokens 31286 -> 27627.
- Mechanism confirmed: fast correct closers appeared (p1 601tok, p2 391, p16 639, p95 508, p112 661 all correct).
- McNemar on correctness b=4/c=6: not significant alone, but closed-rate change is decisive mechanism evidence.
- New closers 9/26 = 35% correct (selection: pass1's 7 closers were the easiest).
- Still 102 cap-hitters, all verbatim loops (median start 16% of generation).

## Levers exhausted (post-pass-2 exploration)
- L0 h8 remaining channels: no silver bullets (only ch48 -1.26 breaks sanity; ch87 -0.51 small).
- h3 total dose <0.5 breaks 17*24 ('360'); h8rest x0.85 breaks 2+2; attn x0.55 breaks 2+2 ('100000000').
- L1h15/L2h5: negligible loop effect.
- Full-attn dose: at x0.2 loop 1.74 but close -2.6 and sanity gone. Edge at ~x0.6+ gives <0.8 loop gain.
- Sink probe: BOS-sink attention evaporates with depth (0.15-0.3 at pos 50 -> 0.000-0.01 at 5990) in most heads;
  zeroing L0 GDN branch removes sinks entirely (L0 creates the anchor signal). No weight lever found to restore persistence.
- Decision: cand_C is at the function edge of the repair family; deeper cuts cost arithmetic/closing.
