"""Sweep MoE-vs-dense branch balance; measure direct answer correctness (greedy)."""
import re

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QS = [
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is $7!$?", "5040"),
    ("Solve for $x$: $2x + 5 = 17$.", "6"),
    ("What is the area of a triangle with base 10 and height 6?", "30"),
    ("If $f(x)=3x-7$, what is $f(5)$?", "8"),
    ("What is $15\\%$ of $80$?", "12"),
    ("Simplify $\\frac{18}{24}$.", "3/4"),
    ("What is $\\sqrt{144}+\\sqrt{25}$?", "17"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

def snap():
    return {k: v.detach().clone() for k, v in lm.state_dict().items()
            if "post_feedforward_layernorm_1" in k or "post_feedforward_layernorm_2" in k}

def restore(s):
    sd = lm.state_dict()
    for k, v in s.items():
        sd[k].copy_(v)

def edit(a, b):
    for layer in lm.layers:
        layer.post_feedforward_layernorm_2.weight.mul_(a)
        layer.post_feedforward_layernorm_1.weight.mul_(b)

def extract(text):
    m = list(re.finditer(r"(-?\d+(?:\.\d+)?)(?:/\d+)?", text))
    boxes = re.findall(r"boxed\{([^}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    return m[-1].group(1) if m else None

def norm_ans(s):
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except Exception:
        return s

CONFIGS = [
    (1.0, 1.0),
    (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (0.5, 1.5), (0.25, 1.5), (0.0, 1.5),
    (0.25, 2.0), (0.0, 2.0),
    (1.5, 1.0),  # opposite direction as control
]

s0 = snap()
for a, b in CONFIGS:
    restore(s0); edit(a, b)
    print(f"\n{'#'*88}\n### moe_x={a} dense_x={b}")
    correct = 0
    for qi, (q, expect) in enumerate(QS):
        enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=300, pad_token_id=0)
        gen = out[0, ids.shape[1]:]
        short = tok.decode(gen, skip_special_tokens=True)
        ans = norm_ans(extract(short))
        ok = ans == norm_ans(expect)
        correct += ok
        print(f"-- Q{qi+1} exp={expect} got={ans!r} ok={ok} :: {short[:200].replace(chr(10),' | ')}")
    print(f"### SCORE {correct}/{len(QS)}")
restore(s0)
print("restored")
