"""Free: batched single-layer single-branch ablation by fitness (one model load).

For each layer L and branch in {attn, mlp, moe}: disable that branch ONLY in L,
measure fitness delta vs baseline. A damaged branch's removal IMPROVES fitness.

Usage: ~/wsp3c-venv/bin/python ablate_batch.py [attn|mlp|moe|all]
"""

import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

TEXT = [
    ("The capital of France is", " Paris, and it is known for the Eiffel Tower."),
    ("Water boils at", " 100 degrees Celsius at sea level."),
    ("The largest planet in our solar system is", " Jupiter, a gas giant."),
    ("17 * 23 =", " 391"),
    ("2 + 2 =", " 4"),
    ("A cat is a small", " domesticated feline animal that is often kept as a pet."),
    ("In mathematics, a prime number is", " a natural number greater than 1 that has no positive divisors other than 1 and itself."),
    ("To convert Celsius to Fahrenheit, multiply by", " 9/5 and add 32."),
    ("The sun rises in the", " east and sets in the west."),
    ("The chemical symbol for gold is", " Au."),
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


def make_ablation(kind, li, layers):
    """Patch layers[li].forward to skip one branch."""
    orig = layers[li].forward

    def fn(hidden_states, per_layer_input=None, **kwargs):
        layer = layers[li]
        residual = hidden_states
        hs = layer.input_layernorm(hidden_states)
        if kind == "attn":
            hs = torch.zeros_like(residual)
        else:
            hs, _ = layer.self_attn(
                hidden_states=hs,
                position_embeddings=kwargs.get("position_embeddings"),
                attention_mask=kwargs.get("attention_mask"),
                shared_kv_states=kwargs.get("shared_kv_states"),
                position_ids=kwargs.get("position_ids"),
                past_key_values=kwargs.get("past_key_values"),
            )
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

    return orig, fn


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    kinds = ["attn", "mlp", "moe"] if which == "all" else [which]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    base = score(model, tok)
    print(f"BASELINE: {base:.4f}")

    results = []
    for kind in kinds:
        print(f"--- no_{kind} per layer ---")
        for li in range(30):
            orig, fn = make_ablation(kind, li, layers)
            layers[li].forward = fn
            try:
                s = score(model, tok)
            finally:
                layers[li].forward = orig
            delta = s - base
            results.append((delta, kind, li))
            flag = ""
            if delta > 0.5:
                flag = "  <== IMPROVES (damaged?)"
            elif delta < -2:
                flag = "  (critical)"
            print(f"  no_{kind}_L{li:02d}: {s:8.4f} delta={delta:+7.4f}{flag}")

    print("\n=== ranked (best deltas first) ===")
    for delta, kind, li in sorted(results, reverse=True)[:10]:
        print(f"  no_{kind}_L{li:02d}: {delta:+7.4f}")


if __name__ == "__main__":
    main()
