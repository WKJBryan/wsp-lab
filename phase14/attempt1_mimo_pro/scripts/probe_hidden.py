"""Probe hidden states for massive activations / outlier dims on a few teacher-forced prompts (CPU)."""
import sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16, device_map='cpu')
model.eval()

prompts = json.load(open(sys.argv[1]))   # list of strings (user messages)
cache = {}

def mk(i):
    def hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cache[i] = o[0].detach().float()   # [T, 1024]
    return hook

lm = model.model if hasattr(model.model, 'layers') else model.model.language_model
hooks = [layer.register_forward_hook(mk(i)) for i, layer in enumerate(lm.layers)]

for p in prompts:
    text = tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)
    ids = tok(text, return_tensors='pt').input_ids
    cache.clear()
    with torch.no_grad():
        model(ids)
    print(f'== prompt: {p[:80]!r} T={ids.shape[1]}')
    for i in range(24):
        h = cache[i]                     # [T, 1024]
        absmax, argmx = h.abs().max(dim=1)          # per position
        row = (f'  L{i:2d} |h| mean={h.norm(dim=1).mean():8.2f} max={h.norm(dim=1).max():8.2f} '
               f'max|dim|={absmax.max():9.2f} at dim {int(argmx[int(absmax.argmax())])}')
        print(row)
    # final norm output
for h in hooks: h.remove()
