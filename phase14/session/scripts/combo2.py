"""Combo screen v2 on cand_C: h8-remainder dose x h3 dose. Loop (10 gen2 problems), close (6 gen2 closers), sanity."""
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
H8 = 8 * HD


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
for r in rows:
    if r['finish'] != 'length' or r['closed']:
        continue
    pid = r['id']
    fl = find_loop(r['text'])
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + r['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
PIDS = list(LOOP)[:10]
for pid in ['p1', 'p2', 'p16', 'p95', 'p112', 'p101']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    if idx >= 0:
        CLOSE[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()

SANity = ['2 + 2 =', 'Q: What is 17 * 24? A:', 'The capital of France is']
SAN_IDS = [tok(p, return_tensors='pt').input_ids.cuda() for p in SANity]

OP = model.model.layers[0].linear_attn.out_proj


def block_hook(blocks):
    """blocks: list of (start, end, scale)"""
    def hook(mod, args):
        x = args[0]
        for s0, s1, sc in blocks:
            x[:, :, s0:s1] = x[:, :, s0:s1] * sc
        return (x,)
    return hook


@torch.no_grad()
def measure(blocks):
    hh = OP.register_forward_pre_hook(block_hook(blocks)) if blocks else None
    lm = {}
    for pid in PIDS:
        logits = model(LOOP[pid]).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        lm[pid] = float(v[0] - v[1])
    cl = {}
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        cl[pid] = float(lp[248069])
    san = []
    for ids in SAN_IDS:
        out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        san.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    if hh:
        hh.remove()
    return lm, cl, san


variants = [
    ('cand_C (base)', None),
    ('h8rest x0.5', [(H8, H8 + HD, 0.5)]),
    ('h8rest x0.25', [(H8, H8 + HD, 0.25)]),
    ('h8rest x0', [(H8, H8 + HD, 0.0)]),
    ('h3 x0.5 (total .25)', [(3 * HD, 4 * HD, 0.5)]),
    ('h3 x0.7 (total .35)', [(3 * HD, 4 * HD, 0.7)]),
    ('h8rest x0.5 + h3 x0.5', [(H8, H8 + HD, 0.5), (3 * HD, 4 * HD, 0.5)]),
    ('h8rest x0.25 + h3 x0.7', [(H8, H8 + HD, 0.25), (3 * HD, 4 * HD, 0.7)]),
]
for name, blocks in variants:
    lm, cl, san = evaluate = measure(blocks)
    mean = sum(lm.values()) / len(lm)
    mx = max(lm.values())
    detail = ' '.join(f'{k}:{v:.1f}' for k, v in lm.items())
    clm = sum(cl.values()) / len(cl)
    print(f'{name:24s} loop mean {mean:5.2f} max {mx:5.2f} [{detail}]')
    print(f'{"":24s} close mean {clm:6.2f} ({[round(v,1) for v in cl.values()]}) san0={san[0][:16]!r} san1={san[1][:22]!r}')
