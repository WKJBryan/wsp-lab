"""Attribute the loop logit: per-layer branch outputs projected onto (top1 - top2) unembed direction."""
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/model_try1'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}
E = model.get_input_embeddings().weight.float()


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


pid = 'p1'
t = byid[pid]['text']
ids = torch.tensor([tok(prompt_for(pid) + t).input_ids[-2200:]]).cuda()
pos = -1  # last position (mid-loop)

cap = {}


def mk_hook(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cap[name] = o[0, pos].detach().float()
    return hook


handles = []
for L in range(24):
    mod = model.model.layers[L]
    attn = mod.self_attn if hasattr(mod, 'self_attn') else mod.linear_attn
    handles.append(attn.register_forward_hook(mk_hook(f'L{L}.attn')))
    handles.append(mod.mlp.register_forward_hook(mk_hook(f'L{L}.mlp')))
with torch.no_grad():
    logits = model(ids).logits[0, pos].float()
for h in handles:
    h.remove()

lp = F.log_softmax(logits, -1)
v, i = torch.topk(lp, 3)
top1, top2 = int(i[0]), int(i[1])
print(f'at loop position: top {tok.decode([top1])!r} {float(v[0]):.2f}, 2nd {tok.decode([top2])!r} {float(v[1]):.2f}')
d = E[top1] - E[top2]
nrm = d.norm()
print(f'branch contributions to logit(top1)-logit(top2) (norm of d = {float(nrm):.1f}):')
vals = []
for name in sorted(cap):
    c = float(cap[name] @ d)
    vals.append((c, name))
for c, name in sorted(vals, key=lambda x: -abs(x[0])):
    print(f'  {name:12s} {c:9.2f}')
