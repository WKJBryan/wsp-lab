"""Decompose L15/19/23 attention by head: which heads write the ' Wait' vs ' So' direction at loop positions?"""
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
NH, HDIM = 8, 256

WAIT = tok.encode(' Wait', add_special_tokens=False)[0]
SO = tok.encode(' So', add_special_tokens=False)[0]
d = (E[WAIT] - E[SO])


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


# per-layer, per-head projection directions: W_h^T d
WD = {}
for L in (15, 19, 23):
    W = model.model.layers[L].self_attn.o_proj.weight.float()  # (1024, 2048)
    WD[L] = torch.stack([W[:, h * HDIM:(h + 1) * HDIM].T @ d for h in range(NH)])  # (8, 256)

cap = {}


def make_pre(L):
    def pre_hook(mod, args):
        x = args[0]
        cap[L] = x[0, -1].detach().float().view(NH, HDIM)
    return pre_hook


handles = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_pre(L)) for L in (15, 19, 23)]

for pid in ['p1', 'p5']:
    t = byid[pid]['text']
    x = torch.tensor([tok(prompt_for(pid) + t).input_ids[-2200:]]).cuda()
    cap.clear()
    with torch.no_grad():
        logits = model(x).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    print(f'=== {pid}: gap(Wait,So) = {float(lp[WAIT] - lp[SO]):.2f}')
    for L in (15, 19, 23):
        heads = cap[L]
        contribs = [(float(heads[h] @ WD[L][h]), h) for h in range(NH)]
        contribs.sort(key=lambda z: -abs(z[0]))
        print(f'  L{L}: ' + ' '.join(f'h{h}:{c:+.2f}' for c, h in contribs))
