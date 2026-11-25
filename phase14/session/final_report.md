# Phase 14 — final report

## Final weights

`~/wsp14/sandbox/model` = **cand_C** (fingerprint `0a4590253bf4b2f4`, accepted by `save_edit.py`).
Identical to `~/wsp14/work/cand_C`; pristine reference preserved at `~/wsp14/work/model_pristine`.

The final edit is a single tensor:

```
model.language_model.layers.0.linear_attn.out_proj.weight   (1024 x 2048, bf16)
  columns [384:512)   (GDN value-head 3)            multiplied by 0.5
  columns 8*128+{44, 50, 76, 81, 120}               set to 0
```

136,192 values changed; largest |delta| 0.084; change norm 2.26. No other tensor was touched.

## Diagnosis of the failure (evidence)

The model has a **verbatim-repetition attractor**: any moderately long generation collapses into a
fixed sentence cycle ("Wait, $728$ is closer to 729." repeated hundreds of times), never emits
`</think>`, and runs to the 32,768-token cap. Baseline (pass 1): accuracy 4.7%, closed 5.5%,
cap-hit 94.5%, mean 31,286 tokens — 0/11 on Level 1.

Evidence trail (teacher-forced reads, all free):

1. All 121 cap-hit generations end in exact verbatim cycles (period 334 chars median); the
   cycle is a *fixed point*: at loop positions the top-1 next token has logprob -0.0 with a
   8-16 nat margin over every alternative — an order of magnitude beyond the protocol's
   presence penalty (1.5), so decoding cannot escape.
2. `</think>` is reachable only from specific "ready" contexts ("The final answer is X.",
   "Okay. Ready.", "This seems correct."): there it sits at logprob ≈ -2 (and becomes top-1
   after the repair). Mid-derivation it is correctly suppressed (-14 to -25). The 7 baseline
   generations that escaped the loop closed and 6/7 were correct — the close-then-answer
   pathway is intact; the loop prevents reaching it.
3. Causal localization: zeroing layer 0's Gated DeltaNet branch (one hook, no weight change)
   collapses the loop prediction (' Wait' -0.0 → '=' -1.7) while no other single layer matters;
   zeroing all six full-attention layers, all 18 GDN layers, or all MLPs each break it too —
   the attractor is a whole-stack resonance seeded by layer 0's readout.
4. Per-head/per-channel maps of layer 0's `out_proj`: head 3 and head 8 carry the loop seed;
   within head 8, channels {44, 50, 76, 81, 120} carry loop-specific signal, while head 8 as a
   whole (its remaining channels; originally the decay≈1.0 "never-forget" head) also carries
   the **closing** signal, and head 3 at doses ≤ 0.5x0.5 carries simple-arithmetic function.
   Closing is carried by exactly two branches: layer-0 GDN and layer-23 MLP (zeroing either
   drops P(</think>) from ≈ -2 to -10..-13).

## What was tried and rejected

- Restoring layer 0 `dt_bias`/`A_log` decay timescales (24 values, to population ranges):
  proxy loop margins unchanged (11.6 → 11.6) — the loop is not state-decay-driven. Dropped.
- Head 3 fully zeroed or x0.25: breaks simple arithmetic ("2 + 2 =" → " 2.").
- Attention-branch doses on the 6 full-attention layers: at x0.2 loop mean falls to 1.7 but
  closing and sanity collapse; useful doses (< x0.6) give < 1 nat of loop margin. Chat-mode
  teacher-forced NLL of real successful generations rises 0.577 → 0.821 (x0.5) → 1.032 (x0.35)
  nats/token — the damage is real in the evaluated path, not a raw-completion artifact.
- L1/L2 per-head surgery (h15/h5): negligible (< 0.8 nat).
- Marginal extra doses (h3 0.45, h8rest 0.85-0.5): ≤ 0.5 nat loop gain, sanity breakage.

## Passes spent (2 of 15)

| pass | candidate | result |
|------|-----------|--------|
| 1 | base model | accuracy 4.7%, closed 5.5%, cap-hit 94.5%, mean 31,286 tok |
| 2 | cand_C | accuracy 7.8%, closed 26.6%, cap-hit 79.7%, mean 27,627 tok |

What each showed: pass 1 established the failure (loop-to-cap, never closing); pass 2 confirmed
the repair mechanism — 5x more generations close their thinking, fast correct closers appeared
(p1/p2/p16/p95/p112, 391-661 tokens, all correct), Level 1 went 0/11 → 3/11. The correctness
delta (+3.1 points) is within one standard error (~4.4) of zero on 128 problems, but the
closed-rate change (7 → 34) is decisive, and closers convert at 35% correct, so the accuracy
direction is supported.

## Candidate refused by the checker

None refused. (Several self-rejected candidates were never submitted: full-tensor scalings are
refused by design and were avoided; h3 zeroed variants were dropped for arithmetic damage.)

## Predicted effect on the held-out set

MATH-500 should improve by roughly **+3 to +8 points** over the broken baseline (4.7% here):
the mechanism gain is large and replicated in-distribution (cap-hit -16 points, closed 5x),
and every converted cap-hitter answers correctly with ~26-35% probability. The absolute score
will stay low (single-digit to low-teens) — the attractor can only be damped, not removed,
without damaging arithmetic and closing (they share the same readout pathways), and a 0.8B
model closing reliably is still bounded by its own math ability.
