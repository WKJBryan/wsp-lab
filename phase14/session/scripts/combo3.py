"""Combo screen v3: cand_C + L1h15 dose + L2h5 dose."""
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


def block_hook(blocks):
    """blocks: list of (layer, start, end, scale)"""
    hooks_spec = blocks

    def hook(mod, args):
        x = args[0]
        for (L, s0, s1, sc) in hooks_spec:
            if mod is model.model.layers[L].linear_attn.out_proj:
                x[:, :, s0:s1] = x[:, :, s0:s1] * sc
        return (x,)
    return hook


@torch.no_grad()
def measure(blocks):
    blocks = blocks or []
    handles = []
    by_layer = {}
    for (L, s0, s1, sc) in blocks:
        by_layer.setdefault(L, []).append((s0, s1, sc))
    for L, specs in by_layer.items():
        OP = model.model.layers[L].linear_attn.out_proj

        def mk(specs):
            def hook(mod, args):
                x = args[0]
                for s0, s1, sc in specs:
                    x[:, :, s0:s1] = x[:, :, s0:s1] * sc
                return (x,)
            return hook
        handles.append(OP.register_forward_pre_hook(mk(specs)))
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
        out = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        san.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    for h in handles:
        h.remove()
    return lm, cl, san


variants = [
    ('cand_C', None),
    ('+L1h15 x0.5', [(1, 15 * HD, 16 * HD, 0.5)]),
    ('+L1h15 x0', [(1, 15 * HD, 16 * HD, 0.0)]),
    ('+L1h15x0 + L2h5x0', [(1, 15 * HD, 16 * HD, 0.0), (2, 5 * HD, 6 * HD, 0.0)]),
    ('+L1h15x0.5 + L2h5x0.5', [(1, 15 * HD, 16 * HD, 0.5), (2, 5 * HD, 6 * HD, 0.5)]),
]
for name, blocks in variants:
    lm, cl, san = measure(blocks)
    mean = sum(lm.values()) / len(lm)
    mx = max(lm.values())
    clm = sum(cl.values()) / len(cl)
    print(f'{name:24s} loop mean {mean:5.2f} max {mx:5.2f} [{" ".join(f"{k}:{v:.1f}" for k,v in lm.items())}]')
    print(f'{"":24s} close mean {clm:6.2f} ({[round(v,1) for v in cl.values()]}) san0={san[0][:16]!r} san1={san[1][:20]!r}')
