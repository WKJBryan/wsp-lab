# The Matrix Atlas — Phase 7 working document

Living document; each entry carries evidence + a confidence label. Direction v = the loop
direction (unit top-singular vector of the L15 edit block; cos(v15,v19)=0.849).

## Finding 1 — the loop direction is written DIFFUSELY, not locally [measured]

Every layer's MLP (down_proj) writes onto v with statistically similar strength:
mean |alignment| ≈ 0.006–0.008, max ≈ 0.05–0.09 across all 24 layers. No hotspot at the
edited layers (15/19 are unremarkable in this metric). The loop direction is a
network-wide residual-stream property, not the product of one layer.

Evidence: `atlas_geometry.json` → down_proj_alignment (full per-layer table).

## Finding 2 — the edited columns were NOT chosen by weight-side alignment [measured]

The 64 columns the original authors edited per layer do not appear among the top-aligned
columns of either edited layer (L15's top-aligned: 1155, 1787, 1060…; their picks: 0, 2, 11,
25, 34…; overlap ≈ none). Their selection must have come from activation-side evidence
(differential activity during loops), not weight geometry. Corollary: weight-geometry alone
cannot rediscover their edit — consistent with our phase-3a blind failure and the field's
no-hint difficulty.

## Finding 3 — attention-side: layer 23 is the strongest writer onto v [measured]

Top layers by max |o_proj alignment| onto v: **L23 (0.134)**, L19 (0.106), L3 (0.085),
L15 (0.083), L7 (0.070). Notable: openwsp's independent novel edit (this week) targeted
layer 23's down_proj — convergence from two directions (ours: geometry; theirs: search).
L23 is a prime candidate for the phase-6 layer-extension family.

## Finding 4 — token map: the −v side is rich in "alternative-continuation" function words [suggestive]

Top −v-aligned embedding rows: " thereby", " rather", " or", " thus", " first",
" otherwise", " easily", " rather than" family; +v side: rare multilingual fragments.
Reading (SPECULATIVE until generation-time probing confirms): suppressing +v /
amplifying −v overlaps the semantics of offering-another-way-to-continue — plausibly the
"Wait, but let me reconsider…" attractor region. Alignments are weak (|a| ≤ 0.10), so this
is a hypothesis, not a claim. Tied lm_head means the same map covers output logits.

## Next (needs GPU, queued behind phase-5 finals)

- Activation probing on stuck-vs-clean generations: which neurons actually fire differently
  (the "light up" map), overlaid on Finding 1's geometry — the key test of Finding 2's
  activation-vs-geometry distinction.
- Generation-time token probe: which tokens' logits move when v is perturbed ±.
- Causal spot-checks on Findings 3 & 4 per protocol.
