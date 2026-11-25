"""Per-channel map of L0 h8: zero each of 128 channels; effect on loop margins (p1,p9) and close P(th)."""
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
H8 = 8 * HD
H3 = 3 * HD


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
for pid in ['p1', 'p9']:
    t = byid[pid]['text']
    fl = find_loop(t)
    CTX[('loop', pid)] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CTX[('close', pid)] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()

OP = model.model.layers[0].linear_attn.out_proj


def ch_hook(channels, scale):
    def hook(mod, args):
        x = args[0]
        for c in channels:
            x[:, :, H8 + c] = x[:, :, H8 + c] * scale
        return (x,)
    return hook


@torch.no_grad()
def measure(channels=None, scale=0.0, extra=None):
    hh = None
    if channels or extra:
        def hook(mod, args):
            x = args[0]
            if channels:
                for c in channels:
                    x[:, :, H8 + c] = x[:, :, H8 + c] * scale
            if extra:
                for h, s in extra.items():
                    x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * s
            return (x,)
        hh = OP.register_forward_pre_hook(hook)
    out = {}
    for k, ids in CTX.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        if k[0] == 'loop':
            v, i = torch.topk(lp, 2)
            out[k] = float(v[0] - v[1])
        else:
            out[k] = float(lp[248069])
    if hh:
        hh.remove()
    return out


base = measure()
print('base: p1', round(base[('loop', 'p1')], 1), 'p9', round(base[('loop', 'p9')], 1),
      'close mean', round(sum(v for k, v in base.items() if k[0] == 'close') / 4, 2))
print()
print('per-channel zeros: dLoop(p1+p9) [negative = loop weaker] | dClose [positive = closing weaker]')
rows_out = []
for c in range(HD):
    r = measure([c])
    dp1 = r[('loop', 'p1')] - base[('loop', 'p1')]
    dp9 = r[('loop', 'p9')] - base[('loop', 'p9')]
    dc = sum(v for k, v in r.items() if k[0] == 'close') / 4 - sum(v for k, v in base.items() if k[0] == 'close') / 4
    rows_out.append((c, dp1, dp9, dc))
for c, dp1, dp9, dc in rows_out:
    flag = ''
    if dp1 < -1 or dp9 < -1:
        flag += ' LOOP'
    if dc > 1:
        flag += ' CLOSE'
    print(f'  ch{c:3d}: dp1 {dp1:+6.2f} dp9 {dp9:+6.2f} dclose {dc:+6.2f}{flag}')
