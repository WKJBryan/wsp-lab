"""Free: bigger fitness set (30 seqs) + family-scale sweep for attention/MLP/expert
matrices (untested before) + validation of top ablation hits + top combos.
One model load.
"""

import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

TEXT = [
    # facts
    ("The capital of France is", " Paris, and it is known for the Eiffel Tower."),
    ("The largest planet in our solar system is", " Jupiter, a gas giant."),
    ("The chemical symbol for gold is", " Au."),
    ("Water boils at", " 100 degrees Celsius at sea level."),
    ("The sun rises in the", " east and sets in the west."),
    ("Shakespeare wrote the play", " Hamlet, among many other tragedies."),
    ("The human heart has", " four chambers that pump blood through the body."),
    ("Photosynthesis takes place in the", " chloroplasts of plant cells."),
    # math
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("100 - 47 =", " 53"),
    ("sqrt(144) =", " 12"),
    ("A train travels 60 mph for 2 hours, covering", " 120 miles in total."),
    ("If x + 7 = 12, then x =", " 5"),
    ("The sum of the first 10 positive integers is", " 55."),
    # prose / reasoning
    ("A cat is a small", " domesticated feline animal that is often kept as a pet."),
    ("In mathematics, a prime number is", " a natural number greater than 1 that has no positive divisors other than 1 and itself."),
    ("To convert Celsius to Fahrenheit, multiply by", " 9/5 and add 32 to the result."),
    ("The mitochondria is the", " powerhouse of the cell, producing ATP energy."),
    ("Gravity is the force that", " attracts objects with mass toward each other."),
    ("Python is a programming language known for its", " simple syntax and readability."),
    ("The Second World War ended in", " 1945 with the surrender of Germany and Japan."),
    ("Oil floats on water because it is", " less dense than water is."),
    ("The boiling point of water at sea level is", " 100 degrees Celsius or 212 Fahrenheit."),
    ("Mount Everest is the", " highest mountain on Earth above sea level."),
    ("Antibiotics are used to treat", " bacterial infections caused by microbes."),
    ("The Pacific Ocean is the", " largest and deepest of Earth's ocean basins."),
    ("DNA stands for", " deoxyribonucleic acid, the molecule carrying genetic code."),
    ("Ice is frozen", " water that has solidified below zero degrees Celsius."),
    ("The speed of light is approximately", " 300,000 kilometers per second."),
]


def score(model, tok):
    total, count = 0.0, 0
    with torch.no_grad():
        for prompt, cont in TEXT:
            p = tok.encode(prompt, add_special_tokens=False)
            c = tok.encode(cont, add_special_tokens=False)
            ids = torch.tensor([p + c]).to(model.device)
            logits = model(ids).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            for i in range(len(p) - 1, len(ids[0]) - 1):
                total += lp[i, ids[0, i + 1]].item()
                count += 1
    return total / count


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lm = model.model.language_model
    layers = lm.layers

    # snapshot everything we may touch
    orig = {"final": lm.norm.weight.data.clone(),
            "embed": lm.embed_tokens.weight.data.clone(),
            "lscalar": torch.stack([l.layer_scalar.clone() for l in layers])}
    for attr in ("post_feedforward_layernorm", "post_attention_layernorm",
                 "input_layernorm", "pre_feedforward_layernorm"):
        orig[attr] = [getattr(l, attr).weight.data.clone() for l in layers]
    for name, mod in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
                      ("o", "o_proj")):
        orig[name] = []
        for l in layers:
            m = getattr(l.self_attn, mod, None)
            orig[name].append(m.weight.data.clone() if m is not None else None)
    orig["gate"] = [l.mlp.gate_proj.weight.data.clone() for l in layers]
    orig["up"] = [l.mlp.up_proj.weight.data.clone() for l in layers]
    orig["down"] = [l.mlp.down_proj.weight.data.clone() for l in layers]
    orig["gu"] = [l.experts.gate_up_proj.data.clone() for l in layers]
    orig["dn"] = [l.experts.down_proj.data.clone() for l in layers]

    def restore():
        lm.norm.weight.data.copy_(orig["final"])
        lm.embed_tokens.weight.data.copy_(orig["embed"])
        for i, l in enumerate(layers):
            l.layer_scalar.copy_(orig["lscalar"][i])
            for attr in ("post_feedforward_layernorm", "post_attention_layernorm",
                         "input_layernorm", "pre_feedforward_layernorm"):
                getattr(l, attr).weight.data.copy_(orig[attr][i])
            for name, mod in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
                              ("o", "o_proj")):
                m = getattr(l.self_attn, mod, None)
                if m is not None and orig[name][i] is not None:
                    m.weight.data.copy_(orig[name][i])
            l.mlp.gate_proj.weight.data.copy_(orig["gate"][i])
            l.mlp.up_proj.weight.data.copy_(orig["up"][i])
            l.mlp.down_proj.weight.data.copy_(orig["down"][i])
            l.experts.gate_up_proj.data.copy_(orig["gu"][i])
            l.experts.down_proj.data.copy_(orig["dn"][i])

    base = score(model, tok)
    print(f"BASELINE (30-seq): {base:.4f}")

    def scale_family(key, c, per_layer=True):
        def f():
            for i, l in enumerate(layers):
                if key == "final":
                    lm.norm.weight.data.mul_(c)
                elif key == "embed":
                    lm.embed_tokens.weight.data.mul_(c)
                elif key in ("q", "k", "v", "o"):
                    mod = {"q": "q_proj", "k": "k_proj", "v": "v_proj",
                           "o": "o_proj"}[key]
                    m = getattr(l.self_attn, mod, None)
                    if m is not None:
                        m.weight.data.mul_(c)
                elif key == "gate":
                    l.mlp.gate_proj.weight.data.mul_(c)
                elif key == "up":
                    l.mlp.up_proj.weight.data.mul_(c)
                elif key == "down":
                    l.mlp.down_proj.weight.data.mul_(c)
                elif key == "gu":
                    l.experts.gate_up_proj.data.mul_(c)
                elif key == "dn":
                    l.experts.down_proj.data.mul_(c)
        return f

    tests = []
    # family scales (untested before for attn/mlp/experts)
    for key in ("q", "k", "v", "o", "gate", "up", "down", "gu", "dn"):
        for c in (0.5, 2.0):
            tests.append((f"{key}_x{c}", scale_family(key, c)))
    # previously promising
    tests.append(("final_x0.5", scale_family("final", 0.5)))
    tests.append(("embed_x0.5", scale_family("embed", 0.5)))

    def postff_x15():
        for l in layers:
            l.post_feedforward_layernorm.weight.data.mul_(1.5)
    tests.append(("postff_x1.5", postff_x15))

    def postattn_x075():
        for l in layers:
            l.post_attention_layernorm.weight.data.mul_(0.75)
    tests.append(("postattn_x0.75", postattn_x075))

    # top ablation validations (MLP L25, attn L5, moe L29)
    def make_ablation(kind, li):
        def f():
            layer = layers[li]
            orig_fn = layer.forward

            def fn(hidden_states, per_layer_input=None, **kw):
                residual = hidden_states
                hs = layer.input_layernorm(hidden_states)
                if kind == "attn":
                    hs = torch.zeros_like(residual)
                else:
                    hs, _ = layer.self_attn(
                        hidden_states=hs,
                        position_embeddings=kw.get("position_embeddings"),
                        attention_mask=kw.get("attention_mask"),
                        shared_kv_states=kw.get("shared_kv_states"),
                        position_ids=kw.get("position_ids"),
                        past_key_values=kw.get("past_key_values"))
                hs = layer.post_attention_layernorm(hs)
                hs = residual + hs
                residual = hs
                hs = layer.pre_feedforward_layernorm(hs)
                if kind == "mlp":
                    hs = torch.zeros_like(residual)
                else:
                    hs = layer.mlp(hs)
                if layer.enable_moe_block:
                    hs1 = layer.post_feedforward_layernorm_1(hs)
                    flat = residual.reshape(-1, residual.shape[-1])
                    if kind == "moe":
                        hs2 = torch.zeros_like(residual)
                    else:
                        _, w, idx = layer.router(flat)
                        hs2 = layer.pre_feedforward_layernorm_2(flat)
                        hs2 = layer.experts(hs2, idx, w)
                        hs2 = hs2.reshape(residual.shape)
                        hs2 = layer.post_feedforward_layernorm_2(hs2)
                    hs = hs1 + hs2
                hs = layer.post_feedforward_layernorm(hs)
                hs = residual + hs
                return hs * layer.layer_scalar
            layer.forward = fn
            ablation_state["restore"] = orig_fn
        return f

    ablation_state = {}

    def run(name, fn):
        restore()
        if "restore" in ablation_state:
            # nothing
            pass
        try:
            fn()
            s = score(model, tok)
        finally:
            # restore layer forward if ablation
            if "restore" in ablation_state:
                layers[ablation_idx[0]].forward = ablation_state.pop("restore")
            restore()
        delta = s - base
        mark = ""
        if delta > 0.4:
            mark = " <== BETTER"
        elif delta < -1.0:
            mark = " (worse)"
        print(f"  {name:24s} {s:9.4f} delta={delta:+8.4f}{mark}")
        return delta

    for name, fn in tests:
        run(name, fn)

    # ablations need idx tracking
    for kind, li in (("mlp", 25), ("attn", 5), ("moe", 29), ("mlp", 21),
                     ("attn", 22), ("moe", 28)):
        ablation_idx = [li]
        run(f"ablate_{kind}_L{li}", make_ablation(kind, li))

    # combos of the most promising single patches
    print("--- combos ---")
    combo_defs = [
        ("final_x0.5+postff1.5", [scale_family("final", 0.5), postff_x15]),
        ("final_x0.5+postattn.75", [scale_family("final", 0.5), postattn_x075]),
        ("final+postff+postattn", [scale_family("final", 0.5), postff_x15,
                                   postattn_x075]),
        ("embed0.5+postff1.5", [scale_family("embed", 0.5), postff_x15]),
        ("embed0.5+postff+postattn", [scale_family("embed", 0.5), postff_x15,
                                      postattn_x075]),
    ]
    for name, fns in combo_defs:
        def combo(fns=fns):
            for f in fns:
                f()
        run(name, combo)

    restore()


if __name__ == "__main__":
    main()
