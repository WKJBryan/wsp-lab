"""Free: branch ablation — disable attention / dense-MLP / MoE across all layers
(or one layer) and score. The damaged branch's removal should IMPROVE scoring.

Usage: ~/wsp3c-venv/bin/python branch_ablate.py <mode>
modes: base, no_moe, no_mlp, no_attn, no_moe_L<l>, no_mlp_L<l>, no_attn_L<l>
"""

import re
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

PROBES = [
    ("What is 17 * 23? Give only the final number.", ["391"], 40),
    ("What is 7 * 8? Give only the final number.", ["56"], 40),
    ("What is 12 + 34? Give only the final number.", ["46"], 40),
    ("What is 2 + 2? Answer with just the number.", ["4"], 40),
    ("What is 100 - 47? Give only the final number.", ["53"], 40),
    ("What is the capital of France?", ["Paris"], 60),
    ("How many days are in a week? Answer with just the number.", ["7"], 40),
    ("Name one primary color.", ["red", "Red", "blue", "Blue", "yellow", "Yellow"], 40),
    ("The secret code is 7411. What is the secret code? Answer with the number only.",
     ["7411"], 40),
    ("What is 9 + 6? Give only the final number.", ["15"], 40),
]


def run_probes(model, tok):
    ok = 0
    det = []
    for q, expected, maxt in PROBES:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=maxt, do_sample=False,
                               pad_token_id=tok.pad_token_id or 0)
        out = tok.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        good = any(e in out for e in expected)
        ok += good
        det.append((q, out.strip()[:70], good))
    return ok, det


def make_hooks(mode, layers):
    """Returns list of (handle,) to register; monkeypatches layer.forward."""
    orig = [l.forward for l in layers]
    idxs = range(len(layers))
    m = re.match(r"no_(moe|mlp|attn)_L(\d+)$", mode)
    if m:
        idxs = [int(m.group(2))]
        kind = m.group(1)
    elif mode == "no_moe":
        kind = "moe"
    elif mode == "no_mlp":
        kind = "mlp"
    elif mode == "no_attn":
        kind = "attn"
    else:
        raise ValueError(mode)

    def wrap(i, kind, orig_fn):
        def fn(hidden_states, per_layer_input=None, **kwargs):
            layer = layers[i]
            # replicate decoder layer forward with one branch disabled
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
            if kind != "mlp":
                hs = layer.mlp(hs)
            else:
                hs = torch.zeros_like(residual)

            if layer.enable_moe_block:
                hs1 = layer.post_feedforward_layernorm_1(hs)
                flat = residual.reshape(-1, residual.shape[-1])
                if kind != "moe":
                    _, top_k_weights, top_k_index = layer.router(flat)
                    hs2 = layer.pre_feedforward_layernorm_2(flat)
                    hs2 = layer.experts(hs2, top_k_index, top_k_weights)
                    hs2 = hs2.reshape(residual.shape)
                    hs2 = layer.post_feedforward_layernorm_2(hs2)
                else:
                    hs2 = torch.zeros_like(residual)
                hs = hs1 + hs2

            hs = layer.post_feedforward_layernorm(hs)
            hs = residual + hs
            hs = hs * layer.layer_scalar
            return hs
        return fn

    for i in idxs:
        layers[i].forward = wrap(i, kind, orig[i])
    return orig, idxs


def restore(orig, layers, idxs):
    for i in idxs:
        layers[i].forward = orig[i]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    print("=== baseline ===")
    b, _ = run_probes(model, tok)
    print(f"baseline: {b}/10")

    if mode == "base":
        return

    orig, idxs = make_hooks(mode, layers)
    try:
        ok, det = run_probes(model, tok)
    finally:
        restore(orig, layers, idxs)
    flag = " <== IMPROVED" if ok > b else (" (worse)" if ok < b else "")
    print(f"{mode}: {ok}/10{flag}")
    if ok != b:
        for q, out, good in det:
            print(f"   [{'OK' if good else '  '}] {q[:45]:47s} -> {out!r}")


if __name__ == "__main__":
    main()
