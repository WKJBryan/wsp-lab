"""Test RoPE-alignment hypotheses: per-head sharpness diagnostics + in-memory row-roll sweep."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

# ---------- diagnostic: per-head max attention prob (last query) across layers ----------
per_layer_maxprob = {}
def mk_hook(L):
    def hook(mod, inp, out):
        w = out[1]
        if w is None: return
        row = w[0, :, -1, :].float()  # [H, K]
        per_layer_maxprob[L] = row.max(dim=-1).values  # per-head max prob
    return hook
for L, layer in enumerate(lm.layers):
    layer.self_attn.register_forward_hook(mk_hook(L))

def run_prompt(q):
    msgs = [{"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return enc["input_ids"].to("cuda")

ids = run_prompt("Compute $17\\times 23 - 11$. Show your work briefly.")
with torch.no_grad():
    model(ids)
print("per-head max attention prob of last query (per layer): min/mean/max over heads")
for L in range(30):
    mp = per_layer_maxprob.get(L)
    if mp is not None:
        print(f"L{L:<2} min={mp.min():.3f} mean={mp.mean():.3f} max={mp.max():.3f}  n_heads>0.5: {(mp>0.5).sum().item()}")

# ---------- roll sweep ----------
QUESTIONS = [
    "Compute $17\\times 23 - 11$.",
    "What is $1+2+3+\\cdots+100$?",
]

def roll_heads(w, shift, n_heads, head_dim):
    """Roll rows of [n_heads*head_dim, in] within each head block."""
    v = w.view(n_heads, head_dim, -1)
    v = torch.roll(v, shifts=shift, dims=1)
    return v.reshape(w.shape)

orig = {}
for L, layer in enumerate(lm.layers):
    orig[(L, "q")] = layer.self_attn.q_proj.weight.detach().clone()
    orig[(L, "k")] = layer.self_attn.k_proj.weight.detach().clone()

def set_roll(mode, shift_q, shift_k):
    for L, layer in enumerate(lm.layers):
        att = layer.self_attn
        q0, k0 = orig[(L, "q")], orig[(L, "k")]
        nq, hd = q0.shape[0], q0.shape[0] // att.config.num_attention_heads if hasattr(att, "config") else None
        # head dims: q rows = num_heads*head_dim; get head_dim from q_norm
        hdim = att.q_norm.weight.numel()
        if mode in ("both", "q"):
            att.q_proj.weight.copy_(roll_heads(q0, shift_q, q0.shape[0] // hdim, hdim))
        if mode in ("both", "k"):
            kh = k0.shape[0] // hdim
            att.k_proj.weight.copy_(roll_heads(k0, shift_k, kh, hdim))

def restore():
    for L, layer in enumerate(lm.layers):
        layer.self_attn.q_proj.weight.copy_(orig[(L, "q")])
        layer.self_attn.k_proj.weight.copy_(orig[(L, "k")])

CONFIGS = [
    ("both", 128, 128), ("both", 1, 1), ("both", 2, 2),
    ("q", 128, 0), ("k", 128, 0), ("both", 1, -1), ("both", 127, 127),
]

def loop_score(text):
    toks_ = tok(text, add_special_tokens=False)["input_ids"]
    if len(toks_) < 24: return 0.0
    seen = set(); dup = 0; n = 0
    for i in range(len(toks_) - 8):
        g = tuple(toks_[i:i+8])
        if g in seen: dup += 1
        seen.add(g); n += 1
    return dup / max(n, 1)

for mode, sq, sk in CONFIGS:
    restore(); set_roll(mode, sq, sk)
    print(f"\n{'#'*90}\n### roll mode={mode} shift_q={sq} shift_k={sk}")
    for qi, q in enumerate(QUESTIONS):
        ids = run_prompt(q)
        with torch.no_grad():
            out = model.generate(ids, do_sample=False, max_new_tokens=150, pad_token_id=0)
        gen = out[0, ids.shape[1]:]
        short = tok.decode(gen, skip_special_tokens=True)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        top = torch.topk(torch.softmax(logits, -1), 3)
        print(f"-- Q{qi+1} loop={loop_score(short):.3f} prefill_top1={tok.decode(top.indices[:1])!r} p={top.values[0]:.3f}")
        print("  ", short[:400].replace("\n", " | "))
restore()
print("\nrestored original weights")
