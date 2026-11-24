"""Greedy generation trace: per-step diagnostics to find where looping starts."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

msgs = [{"role": "user", "content": "What is $1+2+3+\\cdots+100$?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"].to("cuda")

step_stats = []
attn_ent = {}  # layer -> list of entropies per step

lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

# hook attention to record entropy of last query row
import math
def mk_attn_hook(L):
    def hook(mod, inp, out):
        w = out[1]  # [B, H, Q, K]
        if w is None:
            return
        # last query row
        row = w[0, :, -1, :].float()  # [H, K]
        p = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        ent = -(p * (p + 1e-12).log()).sum(-1).mean().item()  # mean over heads
        attn_ent.setdefault(L, []).append(ent)
    return hook

for L, layer in enumerate(lm.layers):
    layer.self_attn.register_forward_hook(mk_attn_hook(L))

cur = ids
gen = []
for step in range(140):
    with torch.no_grad():
        out = model(cur)
    logits = out.logits[0, -1].float()
    top2 = torch.topk(logits, 2)
    nxt = int(top2.indices[0])
    eos_logit = float(logits[1])
    eos_rank = int((logits > logits[1]).sum())
    d = {
        "step": step, "tok": tok.decode([nxt]),
        "top1": float(top2.values[0]), "gap": float(top2.values[0] - top2.values[1]),
        "eos_logit": eos_logit, "eos_rank": eos_rank,
    }
    step_stats.append(d)
    gen.append(nxt)
    cur = torch.cat([cur, torch.tensor([[nxt]], device="cuda")], dim=1)

text = tok.decode(gen)
print("GENERATED:\n", text)
print("\nstep | tok | top1 | gap | eos_rank | ent_L1 | ent_L5 | ent_L11 | ent_L17 | ent_L23 | ent_L29 | ent_mean_all")
ents = {L: attn_ent.get(L, []) for L in range(30)}
for d in step_stats:
    s = d["step"]
    def e(L):
        return f"{ents[L][s]:6.2f}" if s < len(ents.get(L, [])) else "  --  "
    allmean = sum(ents[L][s] for L in range(30) if s < len(ents.get(L, []))) / 30
    print(f"{s:4d} | {d['tok']!r:>8} | {d['top1']:7.2f} | {d['gap']:6.2f} | {d['eos_rank']:7d} | "
          f"{e(1)} | {e(5)} | {e(11)} | {e(17)} | {e(23)} | {e(29)} | {allmean:6.2f}")
