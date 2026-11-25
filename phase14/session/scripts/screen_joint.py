"""Joint dose screens: full-attn x c1, L0-gdn x c2, all-gdn x c3."""
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
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        CTX[('loop', pid)] = byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CTX[('close', pid)] = t[max(0, idx - 2500):idx]
for pid in ['p28', 'p101']:
    CTX[('mid', pid)] = byid[pid]['text'][:int(0.5 * len(byid[pid]['text']))]
CTXIDS = {k: torch.tensor([tok(prompt_for(k[1]) + v).input_ids[-2500:]]).cuda() for k, v in CTX.items()}


def scale_hook(c):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] * c,) + out[1:]
        return out * c
    return hook


@torch.no_grad()
def run(c_attn, c_l0, c_gdn):
    handles = [model.model.layers[L].self_attn.register_forward_hook(scale_hook(c_attn)) for L in FULL]
    if c_l0 is not None:
        handles.append(model.model.layers[0].linear_attn.register_forward_hook(scale_hook(c_l0)))
    if c_gdn is not None:
        for L in GDN:
            handles.append(model.model.layers[L].linear_attn.register_forward_hook(scale_hook(c_gdn)))
    res = {}
    for k in CTXIDS:
        logits = model(CTXIDS[k]).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[k] = ((tok.decode([i[0]]), float(v[0])), float(v[0] - v[1]), float(lp[248069]))
    for h in handles:
        h.remove()
    return res


variants = [
    ('none', 1.0, None, None),
    ('J1 fa.3+l0.3', 0.3, 0.3, None),
    ('J2 fa.3+gdn.5', 0.3, None, 0.5),
    ('J3 fa.5+l0.1', 0.5, 0.1, None),
    ('J4 fa.2+gdn.3', 0.2, None, 0.3),
    ('J5 fa.3+gdn.3', 0.3, None, 0.3),
    ('J6 gdn.5', 1.0, None, 0.5),
    ('J7 fa.4+gdn.4', 0.4, None, 0.4),
]
for name, ca, cl0, cg in variants:
    res = run(ca, cl0, cg)
    lm = [(kk[1], m) for kk, (t0, m, p) in res.items() if kk[0] == 'loop']
    cl = [p for kk, (t0, m, p) in res.items() if kk[0] == 'close']
    mm = [m for kk, (t0, m, p) in res.items() if kk[0] == 'mid']
    detail = ' '.join(f'{i}:{m:.1f}' for i, m in lm)
    print(f'{name:15s} loop mean {sum(m for _, m in lm) / len(lm):5.2f} [{detail}]  close {sum(cl) / len(cl):6.2f}  mid-margin {sum(mm) / len(mm):4.2f}')
