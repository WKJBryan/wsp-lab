"""Remap L0 out_proj per-head carriers on gen_2-derived loop contexts, under cand_C weights."""
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/cand_C'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_2.jsonl')]
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
for r in rows:
    if r['finish'] != 'length' or r['closed']:
        continue
    pid = r['id']
    t = r['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
PIDS = list(LOOP)[:10]
print('loop problems (gen_2):', PIDS)

OP = model.model.layers[0].linear_attn.out_proj


def head_hook(heads, scale):
    def hook(mod, args):
        x = args[0]
        for h in heads:
            x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * scale
        return (x,)
    return hook


@torch.no_grad()
def margins(heads=None, scale=0.0):
    hh = OP.register_forward_pre_hook(head_hook(heads, scale)) if heads else None
    res = {}
    for pid in PIDS:
        logits = model(LOOP[pid]).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        res[pid] = float(v[0] - v[1])
    if hh:
        hh.remove()
    return res


base = margins()
print('base margins: ' + ' '.join(f'{k}:{v:.1f}' for k, v in base.items()))
print()
print('head | ' + ' | '.join(PIDS))
for h in range(16):
    r = margins([h], 0.0)
    print(f'z{h:2d}  | ' + ' | '.join(f'{r[p]:5.1f}' for p in PIDS))
