"""Teacher-forced reads on CPU: hidden states / attention / MLP activations with hooks.
Usage: python tf_read.py <out.json> '<prompt1>' '<prompt2>' ...
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

out_path = sys.argv[1]
prompts = sys.argv[2:]
MD = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32, device_map='cpu')
model.eval()

records = []
for prompt in prompts:
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    ids = tok(text, return_tensors='pt').input_ids
    hs = {}
    def mk(name):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            hs[name] = o.detach().float()
        return hook
    hooks = []
    lm = model.model if hasattr(model.model, "layers") else model.model.language_model
    for i, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(mk(f'L{i}')))
    with torch.no_grad():
        out = model(ids, output_hidden_states=False)
    for h in hooks: h.remove()
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 20)
    top_toks = [(tok.decode([int(i)]), round(float(p), 4)) for p, i in zip(top.values, top.indices)]
    rec = dict(prompt=prompt[:200], n_tok=int(ids.shape[1]), top_next=top_toks,
               layer_norms={k: round(float(v[0, -1].norm()), 3) for k, v in hs.items()})
    records.append(rec)
    print(json.dumps(rec)[:400])

json.dump(records, open(out_path, 'w'), indent=1)
print('saved', out_path)
