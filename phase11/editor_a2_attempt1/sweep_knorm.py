"""Sweep k_norm scale factor in memory; greedy-generate math answers; detect looping."""
import sys

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QUESTIONS = [
    "What is the greatest common divisor of 84 and 144?",
    "Compute $17\\times 23 - 11$.",
    "What is $1+2+3+\\cdots+100$?",
    "A parabola with equation $y=x^2+bx+c$ passes through the points $(2,3)$ and $(4,3)$. What is $c$?",
]
EXPECT = ["12", "380", "5050", "15"]

FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

def loop_score(text):
    toks = tok(text, add_special_tokens=False)["input_ids"]
    if len(toks) < 24:
        return 0.0
    seen = set(); dup = 0; n = 0
    for i in range(len(toks) - 8):
        g = tuple(toks[i:i+8])
        if g in seen: dup += 1
        seen.add(g); n += 1
    return dup / max(n, 1)

orig_knorm = [layer.self_attn.k_norm.weight.detach().clone() for layer in lm.layers]

for fac in FACTORS:
    # apply in-memory edit (exact restore from saved copies afterwards)
    for layer, orig in zip(lm.layers, orig_knorm):
        layer.self_attn.k_norm.weight.copy_(orig * fac)
    print(f"\n{'#'*90}\n### k_norm factor = {fac}")
    for qi, q in enumerate(QUESTIONS):
        msgs = [{"role": "user", "content": q}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=400,
                                 pad_token_id=tok.pad_token_id or 0)
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        eos = "<eos>" in text or "<end_of_turn>" in text
        ls = loop_score(tok.decode(gen, skip_special_tokens=True))
        short = tok.decode(gen, skip_special_tokens=True)
        ok = EXPECT[qi] in short[:600]
        print(f"\n-- Q{qi+1} expect={EXPECT[qi]} eos={eos} loop8gram={ls:.3f} correct-ish={ok}")
        print(short[:700].replace("\n", " | "))
