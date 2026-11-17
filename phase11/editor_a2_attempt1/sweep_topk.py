"""Test sparse-routing hypotheses:
A) monkeypatch router top_k to {8 (baseline), 4, 2, 1} (diagnostic only, not a weight edit)
B) weight emulation: router.proj.weight *= K (softmax -> ~top-1) for K in {8, 32}
Greedy generation, 4 questions, correctness + loop metrics.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
QS = [
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("Solve for $x$: $2x + 5 = 17$.", "6"),
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

def gen(q, n=200):
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

# --- patch router top_k ---
orig_forward = None
def set_topk(k):
    global orig_forward
    import types
    for layer in lm.layers:
        r = layer.router
        if orig_forward is None:
            orig_forward = r.forward
        if not hasattr(r, "_orig_fwd"):
            r._orig_fwd = r.forward
        if k is None:
            r.forward = r._orig_fwd
        else:
            def mk(kk):
                def fwd(hidden_states, _r=r):
                    hn = _r.norm(hidden_states)
                    hn = hn * _r.scale * _r.scalar_root_size
                    scores = _r.proj(hn)
                    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
                    tw, ti = torch.topk(probs, k=kk, dim=-1)
                    tw = tw / tw.sum(-1, keepdim=True)
                    tw = tw * _r.per_expert_scale[ti]
                    return probs, tw, ti
                return fwd
            r.forward = mk(k)

def run(label):
    print(f"\n{'#'*88}\n### {label}")
    ok = 0
    for q, exp in QS:
        t = gen(q)
        ok += exp in t[:400]
        print(f"-- exp={exp} loop={loop_score(t):.3f} ok={exp in t[:400]} :: {t[:280].replace(chr(10),' | ')}")
    print(f"### SCORE {ok}/{len(QS)}")

# A) monkeypatch top-k
for k in [4, 2, 1]:
    set_topk(k)
    run(f"monkeypatch top_k={k}")
set_topk(None)

# B) weight emulation: sharpen softmax
snaps = {k: v.detach().clone() for k, v in lm.state_dict().items() if "router.proj.weight" in k}
def restore_router():
    sd = lm.state_dict()
    for k, v in snaps.items():
        sd[k].copy_(v)
for K in [8.0, 32.0]:
    restore_router()
    for layer in lm.layers:
        layer.router.proj.weight.mul_(K)
    run(f"router.proj x{K} (weight-level ~top-1)")
restore_router()
run("baseline restored (top_k=8, proj x1)")
