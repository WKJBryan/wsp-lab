"""Zero expert index j in ALL layers (per_expert_scale[j]=0), one j at a time; greedy score."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("Solve for $x$: $2x + 5 = 17$.", "6"),
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

def gen(q, n=160):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=0)
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

print("=== baseline ===")
ls = [loop_score(gen(q)) for q, _ in QS]
print(f"baseline loop={sum(ls)/len(ls):.3f}")

for j in range(8):
    restore()
    for layer in lm.layers:
        layer.router.per_expert_scale.data[j] = 0.0
    ls, ex = [], []
    for q, exp in QS:
        t = gen(q)
        ls.append(loop_score(t))
        ex.append(exp in t[:250])
    print(f"expert[{j}] removed: loop={sum(ls)/len(ls):.3f} correct={sum(ex)}")

# bonus: router forced to top-1 behavior via sharpening expert scores is not linear; skip.
restore()
print("restored")
