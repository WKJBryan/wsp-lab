"""Layer-by-layer activation trace with transformers on a short chat prompt."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

msgs = [{"role": "user", "content": "Compute $17\\times 23 - 11$. Show your work briefly."}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"].to("cuda")
print("prompt tokens:", ids.shape[1])

stats = {}
lm = model.language_model if hasattr(model, "language_model") else model.model.language_model
for i, layer in enumerate(lm.layers):
    d = {}
    stats[i] = d

    def mk(store, key, fn):
        def hook(mod, inp, out):
            store[key] = fn(mod, inp, out)
        return hook

    layer.register_forward_pre_hook(lambda m, inp, idx=i: stats[idx].__setitem__("in_norm", inp[0].detach().float().norm(dim=-1).mean().item()))
    layer.register_forward_hook(lambda m, inp, out, idx=i: stats[idx].__setitem__("out_norm", out.detach().float().norm(dim=-1).mean().item()))
    layer.self_attn.register_forward_hook(lambda m, inp, out, idx=i: stats[idx].__setitem__("attn_out_norm", out[0].detach().float().norm(dim=-1).mean().item()))
    layer.mlp.register_forward_hook(lambda m, inp, out, idx=i: stats[idx].__setitem__("mlp_out_norm", out.detach().float().norm(dim=-1).mean().item()))
    layer.experts.register_forward_hook(lambda m, inp, out, idx=i: stats[idx].__setitem__("moe_out_norm", out.detach().float().norm(dim=-1).mean().item()))
    def router_hook(m, inp, out, idx=i):
        probs, w, idxs = out
        stats[idx]["router_p"] = probs.detach().float().mean(0)  # mean prob per expert
        stats[idx]["router_top1"] = float(probs.detach().float().max(-1).values.mean())
    layer.router.register_forward_hook(router_hook)

with torch.no_grad():
    out = model(ids)
logits = out.logits[0, -1].float()
probs = torch.softmax(logits, -1)
top = torch.topk(logits, 10)
print("\nnext-token top10 (prefill):")
for v, i in zip(top.values, top.indices):
    print(f"  {tok.decode([i])!r:>20} logit={v.item():.3f} p={probs[i].item():.4f}")
eos_ids = [1, 106]
for e in eos_ids:
    print(f"eos {e} ({tok.decode([e])!r}) logit={logits[e].item():.3f} rank={int((logits>logits[e]).sum())}")

print("\nlayer |   in_norm |  out_norm | attn_out  | mlp_out   | moe_out   | router_top1_p | router mean p (8 experts)")
for i in range(30):
    d = stats[i]
    ps = d.get("router_p")
    pstr = " ".join(f"{v:.2f}" for v in ps.tolist()) if ps is not None else "-"
    print(f"L{i:<4} | {d['in_norm']:9.3f} | {d['out_norm']:9.3f} | {d['attn_out_norm']:9.3f} | "
          f"{d['mlp_out_norm']:9.3f} | {d['moe_out_norm']:9.3f} | {d.get('router_top1', float('nan')):.4f} | {pstr}")
