"""Per-layer lesion screen: bypass each layer's attention (post_attention_layernorm.weight=0),
greedy-generate on 3 questions, score. Healthy layers: bypass hurts. Toxic layer: bypass helps.
Also bypass variants for MLP branch (post_ff_ln1=0) and MoE branch (post_ff_ln2=0) in a second pass.
"""
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

snaps = {k: v.detach().clone() for k, v in lm.state_dict().items()}
def restore():
    sd = lm.state_dict()
    for k, v in snaps.items():
        sd[k].copy_(v)

def zero(module_key, L):
    sd = lm.state_dict()
    sd[f"layers.{L}.{module_key}"].zero_()

def run_block(block, label):
    print(f"\n=== lesion: {label} ===")
    for L in range(30):
        restore()
        zero(block, L)
        ls = []
        ex = []
        for q, exp in QS:
            t = gen(q)
            ls.append(loop_score(t))
            ex.append(exp in t[:250])
        print(f"L{L:<2} loop={sum(ls)/len(ls):.3f} correct={sum(ex)}  ex_last={ex}")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "attn"
    block = {"attn": "post_attention_layernorm.weight",
             "dense": "post_feedforward_layernorm_1.weight",
             "moe": "post_feedforward_layernorm_2.weight"}[which]
    run_block(block, which)
    restore()
    print("restored")
