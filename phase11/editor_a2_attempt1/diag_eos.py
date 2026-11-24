"""Diagnostics:
1) EOS logit/rank when teacher-forcing a complete correct answer.
2) Are experts within a layer identical? cos(expert_i, expert_j).
3) Per-head sharpness at a retrieval-critical position (teacher-forced solution).
"""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from safetensors import safe_open

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model

q = "Compute $17\\times 23 - 11$."
answer = ("To compute $17 \\times 23 - 11$:\n\n**Step 1: Multiplication.** "
          "$17 \\times 23 = 391$.\n\n**Step 2: Subtraction.** $391 - 11 = 380$.\n\n"
          "The answer is $\\boxed{380}$.")
msgs = [{"role": "user", "content": q}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
pfx = enc["input_ids"]
ans_ids = tok(answer, add_special_tokens=False)["input_ids"]
full = torch.cat([pfx, torch.tensor([ans_ids])], dim=1).to("cuda")

with torch.no_grad():
    out = model(full)
L = out.logits[0].float()  # [T, V]
# next-token logits at each answer position; find rank of EOS (1) and eot (106)
for name, pos in [("last content token", full.shape[1] - 1), ("after \\boxed{380}$.", full.shape[1] - 1)]:
    lg = L[pos]
    r1 = int((lg > lg[1]).sum()); r106 = int((lg > lg[106]).sum())
    top = torch.topk(lg, 5)
    print(f"{name}: eos rank={r1} eot rank={r106} top5={[(tok.decode([i]), round(v.item(),1)) for v,i in zip(top.values, top.indices)]}")

# what does the model WANT to say after the full correct answer?
lg = L[-1]
top = torch.topk(lg, 10)
print("after correct answer, top10:", [(tok.decode([i]), round(v.item(),1)) for v, i in zip(top.values, top.indices)])

# teacher-forced per-token accuracy on the answer (is the model's knowledge there?)
print("\nteacher-forcing the correct answer (logit rank of each true next token):")
ranks = []
for t in range(pfx.shape[1] - 1, full.shape[1] - 1):
    true_tok = int(full[0, t + 1])
    lg = L[t]
    r = int((lg > lg[true_tok]).sum())
    ranks.append((tok.decode([true_tok]), r))
print(ranks)

# per-head sharpness during the forced answer (last position)
per_layer = {}
def mk_hook(Lx):
    def hook(mod, inp, out):
        w = out[1]
        if w is None: return
        row = w[0, :, -1, :].float()
        per_layer[Lx] = row.max(dim=-1).values
    return hook
for Lx, layer in enumerate(lm.layers):
    layer.self_attn.register_forward_hook(mk_hook(Lx))
with torch.no_grad():
    model(full)
print("\nper-head max attn prob at end of correct answer:")
for Lx in range(30):
    mp = per_layer.get(Lx)
    if mp is not None:
        print(f"L{Lx:<2} min={mp.min():.3f} mean={mp.mean():.3f} max={mp.max():.3f} heads>0.5: {(mp>0.5).sum().item()}")

# expert identity check
print("\nexpert cross-cosine (layer 15 sample + max over layers):")
with safe_open(MODEL + "/model.safetensors", framework="pt", device="cpu") as f:
    worst = 0.0
    for L in range(30):
        gu = f.get_tensor(f"model.language_model.layers.{L}.experts.gate_up_proj").float()
        flat = gu.reshape(8, -1)
        flat = flat / flat.norm(dim=1, keepdim=True)
        C = flat @ flat.T
        off = (C.sum() - C.diagonal().sum()) / (8 * 7)
        worst = max(worst, off.max().item())
        if L == 15:
            print("L15 expert cos matrix:")
            print(C.round(3))
    print(f"max off-diagonal expert cos over all layers: {worst:.4f}")
