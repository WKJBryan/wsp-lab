"""Combo screen: h8/h3 doses on loop (all 6 problems) and close contexts."""
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


def head_hook(spec):
    """spec: {head: scale}"""
    def hook(mod, args):
        x = args[0]
        for h, s in spec.items():
            x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * s
        return (x,)
    return hook


@torch.no_grad()
def measure(spec):
    hh = OP.register_forward_pre_hook(head_hook(spec)) if spec else None
    out = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        out[('loop', pid)] = (float(v[0] - v[1]), tok.decode([i[0]]))
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        out[('close', pid)] = (float(lp[248069]),)
    if hh:
        hh.remove()
    return out


variants = [
    ('base', None),
    ('h8x0.7', {8: 0.7}),
    ('h8x0.5', {8: 0.5}),
    ('h8x0.25', {8: 0.25}),
    ('h8x0', {8: 0.0}),
    ('h8x0.5+h3x0', {8: 0.5, 3: 0.0}),
    ('h8x0+h3x0', {8: 0.0, 3: 0.0}),
    ('h8x0.5+h3x0.5', {8: 0.5, 3: 0.5}),
    ('h8x0.25+h3x0+h11x0+h13x0', {8: 0.25, 3: 0.0, 11: 0.0, 13: 0.0}),
]
for name, spec in variants:
    r = measure(spec)
    lms = [(k[1], v[0], v[1]) for k, v in r.items() if k[0] == 'loop']
    cl = [v[0] for k, v in r.items() if k[0] == 'close']
    mean = sum(m for _, m, _ in lms) / len(lms)
    detail = ' '.join(f'{i}:{m:.1f}({t!r})' for i, m, t in lms)
    print(f'{name:26s} loop mean {mean:5.2f} [{detail}]')
    print(f'{"":26s} close mean {sum(cl) / len(cl):6.2f}  ({[round(c, 1) for c in cl]})')
