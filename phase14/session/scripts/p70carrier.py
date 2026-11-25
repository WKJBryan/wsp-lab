"""What carries the p70/p10/p40/p1 loops? Branch-set ablations on those contexts."""
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/model_pristine'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}
FULL = [3, 7, 11, 15, 19, 23]
GDN = [L for L in range(24) if L not in FULL]


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


def find_loop(text):
    n = len(text)
    period = None
    for cand in range(4, 500):
        unit = text[n - cand:]
        if text[n - 2 * cand:n - cand] == unit and text[n - 3 * cand:n - 2 * cand] == unit:
            period = cand
    if period is None:
        return None
    unit = text[n - period:]
    start = n - period
    while start - period >= 0 and text[start - period:start] == unit:
        start -= period
    return start, unit


CTX = {}
for pid in ['p70', 'p10', 'p40', 'p1']:
    t = byid[pid]['text']
    fl = find_loop(t)
    CTX[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()


def zero_hook(mod, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


@torch.no_grad()
def margins(attn_zero=False, gdn_zero=False, mlp_zero=False):
    handles = []
    for L in range(24):
        if attn_zero:
            mod = model.model.layers[L]
            handles.append((mod.self_attn if L in FULL else mod.linear_attn).register_forward_hook(zero_hook))
        if gdn_zero and L not in FULL:
            handles.append(model.model.layers[L].linear_attn.register_forward_hook(zero_hook))
        if mlp_zero:
            handles.append(model.model.layers[L].mlp.register_forward_hook(zero_hook))
    res = {}
    for pid, ids in CTX.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[pid] = (float(v[0] - v[1]), tok.decode([i[0]]))
    for h in handles:
        h.remove()
    return res


for name, kw in [
    ('none', {}),
    ('attn-zero', {'attn_zero': True}),
    ('gdn-zero', {'gdn_zero': True}),
    ('mlp-zero', {'mlp_zero': True}),
    ('attn+gdn', {'attn_zero': True, 'gdn_zero': True}),
]:
    r = margins(**kw)
    print(f'{name:10s}: ' + ' '.join(f'{k}:{v[0]:5.1f}({v[1][:8]!r})' for k, v in r.items()))
