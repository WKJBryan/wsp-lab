"""Per-layer norms of every weight family; print trend + local deviations."""
import json

rows = json.load(open("/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/stats.json"))
by = {r["name"]: r for r in rows}

fams = {
    "q_proj": "self_attn.q_proj.weight", "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight", "o_proj": "self_attn.o_proj.weight",
    "mlp_gate": "mlp.gate_proj.weight", "mlp_up": "mlp.up_proj.weight", "mlp_down": "mlp.down_proj.weight",
    "exp_gu": "experts.gate_up_proj", "exp_dn": "experts.down_proj",
    "router": "router.proj.weight", "router_scale": "router.scale",
    "in_ln": "input_layernorm.weight", "post_attn_ln": "post_attention_layernorm.weight",
    "pre_ff_ln": "pre_feedforward_layernorm.weight", "post_ff_ln": "post_feedforward_layernorm.weight",
    "pre_ff_ln2": "pre_feedforward_layernorm_2.weight", "post_ff_ln1": "post_feedforward_layernorm_1.weight",
    "post_ff_ln2": "post_feedforward_layernorm_2.weight",
}

# per-family per-layer absmean; detect layers deviating from neighbor trend
import statistics
for fam, suffix in fams.items():
    vals = []
    for L in range(30):
        name = f"model.language_model.layers.{L}.{suffix}"
        r = by.get(name)
        vals.append(r["absmean"] if r else None)
    present = [v for v in vals if v is not None]
    med = statistics.median(present)
    out = []
    for L, v in enumerate(vals):
        if v is None:
            out.append(f"L{L}:--")
        else:
            ratio = v / med
            mark = "*" if ratio > 1.25 or ratio < 0.8 else " "
            out.append(f"L{L}:{v:.4f}({ratio:.2f}){mark}")
    print(f"{fam:<14} med={med:.5f}  " + " ".join(out))
