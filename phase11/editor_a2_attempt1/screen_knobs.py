"""Quick behavioral screen of global norm-weight knobs (in-memory, greedy, 3 questions)."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

QUESTIONS = [
    "Compute $17\\times 23 - 11$.",
    "What is $1+2+3+\\cdots+100$?",
    "What is the greatest common divisor of 84 and 144?",
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
    return {k: v.detach().clone() for k, v in lm.state_dict().items() if any(
        s in k for s in ["router.scale", "post_attention_layernorm.weight", "post_feedforward_layernorm.weight",
                         "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
                         "pre_feedforward_layernorm.weight", "k_norm.weight"])}

def restore(s):
    sd = lm.state_dict()
    for k, v in s.items():
        sd[k].copy_(v)

def edit(fn):
    for L, layer in enumerate(lm.layers):
        fn(L, layer)

CONFIGS = [
    ("baseline", None),
    ("router_scale_x2", lambda L, ly: ly.router.scale.mul_(2.0)),
    ("router_scale_x4", lambda L, ly: ly.router.scale.mul_(4.0)),
    ("post_attn_x0.75", lambda L, ly: ly.post_attention_layernorm.weight.mul_(0.75)),
    ("post_attn_x1.33", lambda L, ly: ly.post_attention_layernorm.weight.mul_(4/3)),
    ("post_ff_x0.75", lambda L, ly: ly.post_feedforward_layernorm.weight.mul_(0.75)),
    ("post_ff_x1.33", lambda L, ly: ly.post_feedforward_layernorm.weight.mul_(4/3)),
    ("moe0.5_dense1.5", lambda L, ly: (ly.post_feedforward_layernorm_2.weight.mul_(0.5), ly.post_feedforward_layernorm_1.weight.mul_(1.5))),
    ("pre_ff_x0.5", lambda L, ly: ly.pre_feedforward_layernorm.weight.mul_(0.5)),
    ("k_norm_x0.8", lambda L, ly: ly.self_attn.k_norm.weight.mul_(0.8)),
]

def loop_score(text):
    t = tok(text, add_special_tokens=False)["input_ids"]
    if len(t) < 24: return 0.0
    seen = set(); dup = 0; n = 0
    for i in range(len(t) - 8):
        g = tuple(t[i:i+8])
        if g in seen: dup += 1
        seen.add(g); n += 1
    return dup / max(n, 1)

s0 = snap()
for name, fn in CONFIGS:
    restore(s0)
    if fn: edit(fn)
    print(f"\n{'#'*88}\n### {name}")
    for qi, q in enumerate(QUESTIONS):
        enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=220, pad_token_id=0)
        gen = out[0, ids.shape[1]:]
        short = tok.decode(gen, skip_special_tokens=True)
        print(f"-- Q{qi+1} loop={loop_score(short):.3f} :: {short[:320].replace(chr(10),' | ')}")
restore(s0)
print("\nrestored")
