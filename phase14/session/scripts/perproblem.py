"""Per-problem loop margins for single-head zeros + mid-context closing check."""
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


LOOP = {}
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40', 'p3', 'p10', 'p20', 'p30', 'p50', 'p70']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
PIDS = list(LOOP)
print('loop problems:', PIDS)

# mid contexts for premature-closing check
MID = {}
for pid in ['p28', 'p101', 'p103', 'p105', 'p27', 'p1']:
    t = byid[pid]['text']
    MID[pid] = torch.tensor([tok(prompt_for(pid) + t[:int(0.5 * len(t))]).input_ids[-2000:]]).cuda()

OP = model.model.layers[0].linear_attn.out_proj


def head_hook(spec):
    def hook(mod, args):
        x = args[0]
        for h, s in spec.items():
            x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * s
        return (x,)
    return hook


@torch.no_grad()
def measure(spec):
    hh = OP.register_forward_pre_hook(head_hook(spec)) if spec else None
    res = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[pid] = float(v[0] - v[1])
    mids = {}
    for pid, ids in MID.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        mids[pid] = float(lp[248069])
    if hh:
        hh.remove()
    return res, mids


base, mid_base = measure(None)
print('base margins: ' + ' '.join(f'{p}:{m:.1f}' for p, m in base.items()))
print('base mid P(th): ' + ' '.join(f'{p}:{v:.1f}' for p, v in mid_base.items()))
print()
hdr = 'head | ' + ' | '.join(PIDS)
print(hdr)
for h in range(16):
    r, _ = measure({h: 0.0})
    print(f'z{h:2d}  | ' + ' | '.join(f'{r[p]:5.1f}' for p in PIDS))
