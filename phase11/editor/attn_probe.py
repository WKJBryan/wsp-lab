"""Attention-map inspection (free, GPU). Checks whether attention is
uniform (k_gain too small), random (RoPE broken), or sane.

Usage (in WSL): ~/wsp3c-venv/bin/python attn_probe.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/user/wsp11/sandbox/model"

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
model.eval()

p = "Remember this secret number: 137.\n\nQuestion: What was the secret number?\nAnswer:"
ids = tok(p, return_tensors="pt")["input_ids"].to("cuda:0")
tokens = tok.convert_ids_to_tokens(ids[0])
print("tokens:", tokens)

with torch.no_grad():
    out = model(ids, output_attentions=True)

atts = out.attentions  # tuple (layers) of (B, heads, q, k)
n_layers = len(atts)
print("n layers with attention:", n_layers)

# positions of the '137' tokens
pos137 = [i for i, t in enumerate(tokens) if t in ("137", "1", "37", "▁137")]
print("candidate answer positions:", pos137)

L = ids.shape[1]
for li, a in enumerate(atts):
    w = a[0].float()  # (heads, L, L)
    last = w[:, -1, :]  # (heads, L) attention from last query
    # entropy over positions
    ent = -(last * (last + 1e-12).log()).sum(-1)
    amax = last.argmax(-1)
    mass_on_137 = last[:, pos137].sum(-1) if pos137 else torch.zeros(last.shape[0])
    mass_recent = last[:, -5:].sum(-1)
    # pre-softmax logit scale proxy: max weight
    print(f"L{li:>2}: heads={w.shape[0]} last-tok attn: "
          f"entropy(mean)={ent.mean():.2f}/{torch.log(torch.tensor(float(L))):.2f} "
          f"argmax(pos)={amax.tolist()[:8]} "
          f"mass@137={mass_on_137.mean():.4f} mass@recent5={mass_recent.mean():.4f} "
          f"maxw={last.max():.3f}")
