"""Screen composite: h3 block x0 + h8 channel subset x0 (variants), all loop problems + close + mid."""
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


LOOP, CLOSE, MID = {}, {}, {}
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40', 'p3', 'p10', 'p20', 'p30', 'p50', 'p70']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CLOSE[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()
for pid in ['p28', 'p101', 'p103', 'p105', 'p27', 'p1']:
    t = byid[pid]['text']
    MID[pid] = torch.tensor([tok(prompt_for(pid) + t[:int(0.5 * len(t))]).input_ids[-2000:]]).cuda()

OP = model.model.layers[0].linear_attn.out_proj


def spec_hook(channels):
    def hook(mod, args):
        x = args[0]
        for c in channels:
            x[:, :, c] = x[:, :, c] * 0.0
        return (x,)
    return hook


@torch.no_grad()
def measure(channels):
    hh = OP.register_forward_pre_hook(spec_hook(channels)) if channels else None
    res = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[('loop', pid)] = (float(v[0] - v[1]), tok.decode([i[0]]))
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        res[('close', pid)] = float(lp[248069])
    for pid, ids in MID.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        res[('mid', pid)] = float(lp[248069])
    if hh:
        hh.remove()
    return res


H3 = [3 * HD + c for c in range(HD)]
H8c = lambda c: 8 * HD + c

variants = {
    'base': [],
    'C1 h3 + ch44,76,120': H3 + [H8c(44), H8c(76), H8c(120)],
    'C2 h3 + ch44,50,76,81,120': H3 + [H8c(44), H8c(50), H8c(76), H8c(81), H8c(120)],
    'C3 h3 + ch44,50,76,81,120,48,56': H3 + [H8c(44), H8c(50), H8c(76), H8c(81), H8c(120), H8c(48), H8c(56)],
    'C4 C3 + ch34,36,60,70': H3 + [H8c(c) for c in (44, 50, 76, 81, 120, 48, 56, 34, 36, 60, 70)],
}
for name, chans in variants.items():
    r = measure(chans)
    lm = [(k[1], v[0], v[1]) for k, v in r.items() if k[0] == 'loop']
    cl = [v for k, v in r.items() if k[0] == 'close']
    md = {k[1]: v for k, v in r.items() if k[0] == 'mid'}
    mean = sum(m for _, m, _ in lm) / len(lm)
    detail = ' '.join(f'{i}:{m:.1f}' for i, m, _ in lm)
    print(f'{name:30s} loop mean {mean:5.2f} [{detail}]')
    print(f'{"":30s} close {[round(c, 1) for c in cl]} mean {sum(cl) / len(cl):5.2f} | mid P(th) {" ".join(f"{k}:{v:.1f}" for k, v in md.items())}')
