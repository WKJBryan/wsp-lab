"""Test: restore layer_scalar to 1.0 (the code's init/trained default) in all layers."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("Solve for $x$: $2x + 5 = 17$.", "6"),
    ("What is the capital of France? Answer in one word.", "Paris"),
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

snaps = {k: v.detach().clone() for k, v in lm.state_dict().items()}
def restore():
    sd = lm.state_dict()
    for k, v in snaps.items():
        sd[k].copy_(v)

def gen(q, n=250, sampler="greedy"):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=(sampler == "sample"), max_new_tokens=n,
                             pad_token_id=0, temperature=1.0, top_p=0.95, top_k=64)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def loop_score(text):
    t = tok(text, add_special_tokens=False)["input_ids"]
    if len(t) < 24: return 0.0
    seen = set(); dup = 0; n = 0
    for i in range(len(t) - 8):
        g = tuple(t[i:i+8])
        if g in seen: dup += 1
        seen.add(g); n += 1
    return dup / max(n, 1)

def run(label):
    print(f"\n{'#'*88}\n### {label}")
    ok = 0
    for q, exp in QS:
        t = gen(q)
        okk = exp in t[:400]
        ok += okk
        print(f"-- exp={exp} loop={loop_score(t):.3f} ok={okk} :: {t[:300].replace(chr(10),' | ')}")
    print(f"### SCORE {ok}/{len(QS)}")

run("baseline (stored layer_scalars)")
restore()
for L, layer in enumerate(lm.layers):
    layer.layer_scalar.fill_(1.0)
run("layer_scalar = 1.0 everywhere")
restore()
print("restored")
