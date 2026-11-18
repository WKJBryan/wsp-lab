"""Free in-memory patch harness: load model once, run fixed probe set, apply
candidate weight patches, re-run, report deltas. Never run during an official pass.

Usage:
  ~/wsp3c-venv/bin/python harness.py baseline
  ~/wsp3c-venv/bin/python harness.py patch1 patch2 ...
"""

import re
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

# (prompt, expected-substring(s) any-of, greedy max new tokens)
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

# ---------- patches (each: (name, fn(model)->None), reversible via saved clones) ----------

def p_layer_scalar_ones(model):
    for layer in model.model.language_model.layers:
        layer.layer_scalar.fill_(1.0)


def p_l0_norms_family(model):
    """Replace L0's two outlier norms with L1's tensors (family-typical shape)."""
    lm = model.model.language_model.layers
    for fam in ("post_attention_layernorm.weight", "post_feedforward_layernorm.weight"):
        getattr(lm[0], fam.split(".")[0]).weight.data.copy_(
            getattr(lm[1], fam.split(".")[0]).weight.data)


def p_postff_ones(model):
    for layer in model.model.language_model.layers:
        layer.post_feedforward_layernorm.weight.data.fill_(1.0)


def p_postff_mean(model):
    for layer in model.model.language_model.layers:
        w = layer.post_feedforward_layernorm.weight.data
        w.fill_(w.mean())


def p_l0_scale_down(model):
    """Scale L0's two outlier norms so their rms matches L1's."""
    lm = model.model.language_model.layers
    for attr in ("post_attention_layernorm", "post_feedforward_layernorm"):
        w0 = getattr(lm[0], attr).weight.data
        w1 = getattr(lm[1], attr).weight.data
        r0 = w0.float().pow(2).mean().sqrt()
        r1 = w1.float().pow(2).mean().sqrt()
        w0.mul_(r1 / r0)


def p_final_norm_clip(model):
    w = model.model.language_model.norm.weight.data
    w.clamp_(max=50.0)


def p_final_norm_ones(model):
    model.model.language_model.norm.weight.data.fill_(1.0)


def p_router_scale_ones(model):
    for layer in model.model.language_model.layers:
        layer.router.scale.data.fill_(1.0)


def p_qk_norm_ones(model):
    for layer in model.model.language_model.layers:
        if layer.self_attn.q_norm.weight is not None:
            layer.self_attn.q_norm.weight.data.fill_(1.0)
        if getattr(layer.self_attn.k_norm, "weight", None) is not None:
            layer.self_attn.k_norm.weight.data.fill_(1.0)


PATCHES = {
    "layer_scalar_ones": p_layer_scalar_ones,
    "l0_norms_family": p_l0_norms_family,
    "postff_ones": p_postff_ones,
    "postff_mean": p_postff_mean,
    "l0_scale_down": p_l0_scale_down,
    "final_clip50": p_final_norm_clip,
    "final_ones": p_final_norm_ones,
    "router_ones": p_router_scale_ones,
    "qk_ones": p_qk_norm_ones,
}


def detect_loop(text):
    """True if some 6-token sequence repeats >=3 times consecutively (roughly)."""
    toks = text.split()
    n = len(toks)
    for k in range(3, min(12, n // 3)):
        for i in range(n - 3 * k):
            seg = toks[i:i + k]
            if seg == toks[i + k:i + 2 * k] == toks[i + 2 * k:i + 3 * k]:
                return True
    # char-level fallback for subword loops
    for k in range(4, 30):
        cnt = 0
        for m in re.finditer(r"(?s)(.{%d})\1\1" % k, text):
            cnt += 1
            if cnt >= 2:
                return True
    return False


def run_probes(model, tok, greedy_only=True):
    results = []
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
        ok = any(e in out for e in expected)
        loop = detect_loop(out)
        results.append((q, out.strip().replace("\n", " | ")[:90], ok, loop))
    return results


def score(results):
    return sum(1 for _, _, ok, _ in results if ok), sum(1 for _, _, _, lp in results if lp)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    print("=== BASELINE ===")
    base = run_probes(model, tok)
    for q, out, ok, loop in base:
        print(f"  [{'OK ' if ok else 'BAD'}{' LOOP' if loop else '     '}] {q[:50]:52s} -> {out!r}")
    b_ok, b_loop = score(base)
    print(f"  score: {b_ok}/{len(base)} correct, {b_loop} loops")

    if mode == "baseline":
        return

    # save originals of tensors the requested patches touch
    orig = {}
    lm = model.model.language_model.layers

    def snapshot():
        s = {"layer_scalar": torch.stack([l.layer_scalar.clone() for l in lm]),
             "post_attn": [l.post_attention_layernorm.weight.data.clone() for l in lm],
             "post_ff": [l.post_feedforward_layernorm.weight.data.clone() for l in lm],
             "l0_post_attn": lm[0].post_attention_layernorm.weight.data.clone(),
             "l0_post_ff": lm[0].post_feedforward_layernorm.weight.data.clone(),
             "final": model.model.language_model.norm.weight.data.clone(),
             "router_scale": [l.router.scale.data.clone() for l in lm],
             "qnorm": [l.self_attn.q_norm.weight.data.clone() for l in lm],
             "knorm": [l.self_attn.k_norm.weight.data.clone() for l in lm]}
        return s

    orig = snapshot()

    def restore():
        for i, l in enumerate(lm):
            l.layer_scalar.copy_(orig["layer_scalar"][i])
            l.post_attention_layernorm.weight.data.copy_(orig["post_attn"][i])
            l.post_feedforward_layernorm.weight.data.copy_(orig["post_ff"][i])
            l.router.scale.data.copy_(orig["router_scale"][i])
            l.self_attn.q_norm.weight.data.copy_(orig["qnorm"][i])
            l.self_attn.k_norm.weight.data.copy_(orig["knorm"][i])
        lm[0].post_attention_layernorm.weight.data.copy_(orig["l0_post_attn"])
        lm[0].post_feedforward_layernorm.weight.data.copy_(orig["l0_post_ff"])
        model.model.language_model.norm.weight.data.copy_(orig["final"])

    names = sys.argv[1:]
    for name in names:
        if name not in PATCHES:
            print(f"unknown patch: {name}")
            continue
        restore()
        PATCHES[name](model)
        res = run_probes(model, tok)
        ok, loops = score(res)
        print(f"=== PATCH {name}: {ok}/{len(res)} correct, {loops} loops "
              f"(baseline {b_ok}/{len(base)} correct, {b_loop} loops) ===")
        for (q, out, okk, loop), (bq, bout, bok, bloop) in zip(res, base):
            mark = ""
            if okk != bok:
                mark += " +RIGHT" if okk else " -RIGHT"
            if loop != bloop:
                mark += " +LOOP" if loop else " -LOOP"
            if mark:
                print(f"   {mark} {q[:45]:47s} -> {out!r}")
    restore()


if __name__ == "__main__":
    main()
