"""Test head-block roll/reversal permutations of attention projections (undo a head-order sabotage)."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

snaps = {k: v.detach().clone() for k, v in lm.state_dict().items()
         if any(s in k for s in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"])}
def restore():
    sd = lm.state_dict()
    for k, v in snaps.items():
        sd[k].copy_(v)

def roll_rows_heads(L, which, shift, reverse=False):
    att = lm.layers[L].self_attn
    W = getattr(att, f"{which}_proj").weight.data
    hdim = att.q_norm.weight.numel()
    nh = W.shape[0] // hdim
    v = W.view(nh, hdim, W.shape[1])
    if reverse:
        v = v.flip(0)
    elif shift:
        v = torch.roll(v, shifts=shift, dims=0)
    getattr(att, f"{which}_proj").weight.data.copy_(v.reshape(W.shape))

def roll_cols_heads(L, shift, reverse=False):
    """o_proj: input dim is heads; roll the INPUT head blocks (columns)."""
    att = lm.layers[L].self_attn
    W = att.o_proj.weight.data  # [hidden, heads*hdim]
    hdim = att.q_norm.weight.numel()
    nh = W.shape[1] // hdim
    v = W.view(W.shape[0], nh, hdim)
    if reverse:
        v = v.flip(1)
    elif shift:
        v = torch.roll(v, shifts=shift, dims=1)
    att.o_proj.weight.data.copy_(v.reshape(W.shape))

def gen(q, n=140):
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=n, pad_token_id=0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def run(label):
    ok = 0
    outs = []
    for q, exp in QS:
        t = gen(q)
        okk = exp in t[:300]
        ok += okk
        outs.append(f"exp={exp} ok={okk} :: {t[:110].replace(chr(10),' | ')}")
    print(f"### {label}: SCORE {ok}/{len(QS)}")
    for o in outs:
        print("   ", o)

CONFIGS = [
    ("o_cols_roll+1", lambda: [roll_cols_heads(L, 1) for L in range(30)]),
    ("o_cols_roll-1", lambda: [roll_cols_heads(L, -1) for L in range(30)]),
    ("o_cols_rev", lambda: [roll_cols_heads(L, 0, reverse=True) for L in range(30)]),
    ("q_roll+1", lambda: [roll_rows_heads(L, "q", 1) for L in range(30)]),
    ("k_roll+1", lambda: [roll_rows_heads(L, "k", 1) for L in range(30)]),
    ("v_roll+1", lambda: [roll_rows_heads(L, "v", 1) for L in range(30)]),
    ("qk_roll+1", lambda: ([roll_rows_heads(L, "q", 1) for L in range(30)],
                           [roll_rows_heads(L, "k", 1) for L in range(30)])),
    ("o_cols_half_swap", lambda: [roll_cols_heads(L, 8) for L in range(30)]),
]

for label, fn in CONFIGS:
    restore()
    fn()
    run(label)
restore()
print("restored")
