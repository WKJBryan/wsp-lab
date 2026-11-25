"""Dose curve: h3 scale x {1, .75, .5, .25, .1, 0} with/without h8 surgical channels. Loop + close + sanity."""
import json, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/model_pristine'
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
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40', 'p3', 'p10', 'p20', 'p30', 'p50', 'p70']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CLOSE[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()

SANity_PROMPTS = ['2 + 2 =', 'Q: What is 17 * 24? A:', 'The capital of France is']
SANITY_IDS = [tok(p, return_tensors='pt').input_ids.cuda() for p in SANity_PROMPTS]

OP = model.model.layers[0].linear_attn.out_proj


def block_hook(spec):
    """spec: list of (start_col, end_col, scale)"""
    def hook(mod, args):
        x = args[0]
        for s0, s1, sc in spec:
            x[:, :, s0:s1] = x[:, :, s0:s1] * sc
        return (x,)
    return hook


@torch.no_grad()
def evaluate(spec):
    hh = OP.register_forward_pre_hook(block_hook(spec)) if spec else None
    lm = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        lm[pid] = float(v[0] - v[1])
    cl = []
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        cl.append(float(lp[248069]))
    san = []
    for ids in SANITY_IDS:
        out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        san.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    if hh:
        hh.remove()
    return lm, cl, san


H3 = (3 * HD, 4 * HD)
H8c = lambda c: (8 * HD + c, 8 * HD + c + 1)
H8CH = [44, 50, 76, 81, 120]

for name, spec in [
    ('base', None),
    ('h8ch only', [H8c(c) + (0.0,) for c in H8CH]),
    ('h3x0.5', [H3 + (0.5,)]),
    ('h3x0.5+h8ch', [H3 + (0.5,)] + [H8c(c) + (0.0,) for c in H8CH]),
    ('h3x0.25+h8ch', [H3 + (0.25,)] + [H8c(c) + (0.0,) for c in H8CH]),
    ('h3x0.1+h8ch', [H3 + (0.1,)] + [H8c(c) + (0.0,) for c in H8CH]),
    ('h3x0+h8ch', [H3 + (0.0,)] + [H8c(c) + (0.0,) for c in H8CH]),
    ('h3x0.75+h8ch', [H3 + (0.75,)] + [H8c(c) + (0.0,) for c in H8CH]),
]:
    lm, cl, san = evaluate(spec)
    mean = sum(lm.values()) / len(lm)
    mx = max(lm.values())
    detail = ' '.join(f'{k}:{v:.1f}' for k, v in lm.items())
    print(f'{name:14s} loop mean {mean:5.2f} max {mx:5.2f} [{detail}]')
    print(f'{"":14s} close {[round(c, 1) for c in cl]} san0={san[0][:20]!r} san1={san[1][:28]!r} san2={san[2][:20]!r}')
