"""Free: localize damage by per-layer bypass. For each layer l, replace its
output with its input (identity) and score the probe set. A damaged layer's
removal should IMPROVE the score.

Usage: ~/wsp3c-venv/bin/python bypass_scan.py [start] [count]
"""

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
    details = []
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
        details.append((q, out.strip()[:60], good))
    return ok, details


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers

    print("=== baseline ===")
    b_ok, _ = run_probes(model, tok)
    print(f"baseline: {b_ok}/10")

    orig = [l.forward for l in layers]

    for i in range(start, min(start + count, len(layers))):
        def make_identity(l_idx):
            def fn(hidden_states, *args, **kwargs):
                return layers[l_idx].layer_scalar * hidden_states if False else hidden_states
            return fn

        layers[i].forward = make_identity(i)
        try:
            ok, details = run_probes(model, tok)
        finally:
            layers[i].forward = orig[i]
        flag = ""
        if ok > b_ok:
            flag = f"  <== IMPROVED by bypassing L{i}"
        elif ok < b_ok:
            flag = f"  (worse)"
        print(f"bypass L{i:02d}: {ok}/10{flag}")
        if ok > b_ok:
            for q, out, good in details:
                print(f"    [{'OK' if good else '  '}] {q[:45]:47s} -> {out!r}")


if __name__ == "__main__":
    main()
