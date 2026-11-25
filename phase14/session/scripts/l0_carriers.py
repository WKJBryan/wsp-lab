"""L0 out_proj per-head carriers: closing (close contexts) vs loop (loop contexts)."""
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
HD = 128


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


LOOP, CLOSE = {}, {}
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CLOSE[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()

OP = model.model.layers[0].linear_attn.out_proj


def head_hook(heads, scale):
    def hook(mod, args):
        x = args[0]
        for h in heads:
            x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * scale
        return (x,)
    return hook


@torch.no_grad()
def measure(heads, scale):
    h = OP.register_forward_pre_hook(head_hook(heads, scale)) if heads else None
    out = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        out[('loop', pid)] = (float(v[0] - v[1]), tok.decode([i[0]]))
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        out[('close', pid)] = (float(lp[248069]), tok.decode([torch.argmax(logits)]))
    if h:
        h.remove()
    return out


base = measure(None, 1.0)
print('base loop margins:', {k[1]: round(v[0], 1) for k, v in base.items() if k[0] == 'loop'})
print('base close P(th): ', {k[1]: round(v[0], 1) for k, v in base.items() if k[0] == 'close'})
print()
print('per-head zero: loop margin mean | close P(th) mean | top-1 on p1-loop, p124-close')
for h in range(16):
    r = measure([h], 0.0)
    lm = sum(v[0] for k, v in r.items() if k[0] == 'loop') / 6
    cl = sum(v[0] for k, v in r.items() if k[0] == 'close') / 4
    print(f'  h{h:2d}: loop {lm:6.2f}  close {cl:7.2f}   p1top {r[("loop","p1")][1]!r}  p124top {r[("close","p124")][1]!r}')
