"""Dose-response: scale all 6 full-attn attention branches by c; measure loop margins, close P(th)."""
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
CTXIDS = {k: torch.tensor([tok(prompt_for(k[1]) + v).input_ids[-2500:]]).cuda() for k, v in CTX.items()}


def scale_hook(c):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] * c,) + out[1:]
        return out * c
    return hook


@torch.no_grad()
def run(c):
    handles = [model.model.layers[L].self_attn.register_forward_hook(scale_hook(c)) for L in FULL]
    res = {}
    for k in CTXIDS:
        logits = model(CTXIDS[k]).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[k] = ((tok.decode([i[0]]), float(v[0])), float(v[0] - v[1]), float(lp[248069]))
    for h in handles:
        h.remove()
    return res


for c in (1.0, 0.75, 0.5, 0.35, 0.2, 0.0):
    res = run(c)
    lm = [m for kk, (t0, m, p) in res.items() if kk[0] == 'loop']
    cl = [p for kk, (t0, m, p) in res.items() if kk[0] == 'close']
    detail = ' '.join(f'{kk[1]}:{m:.1f}' for kk, (t0, m, p) in res.items() if kk[0] == 'loop')
    print(f'attn x{c:4.2f}: loop margins {detail} | mean {sum(lm) / len(lm):5.2f} | close P(th) mean {sum(cl) / len(cl):6.2f}')
