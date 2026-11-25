"""1) Truncation test: does the loop survive on short context windows?
2) Close contexts under full layer-0 GDN branch zeroing (hook).
"""
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


print('=== truncation: loop margin with only the last N tokens of loop text ===')
for pid in ['p1', 'p5']:
    t = byid[pid]['text']
    start, unit = find_loop(t)
    tail = t[start:]  # from loop start to end
    tail_ids = tok(tail).input_ids
    p_ids = tok(prompt_for(pid)).input_ids
    print(f'--- {pid} (loop unit {len(tok(unit).input_ids)} tokens, loop text {len(tail_ids)} tokens)')
    for N in (16, 32, 64, 128, 256, 512):
        ids = torch.tensor([p_ids + tail_ids[-N:]]).cuda()
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 3)
        print(f'   N={N:4d}: top {tok.decode([i[0]])!r} margin {float(v[0] - v[1]):6.2f} '
              f'2nd {tok.decode([i[1]])!r} {float(v[1]):6.2f}')

print()
print('=== close contexts with layer-0 GDN branch fully zeroed (hook) ===')
CTX = {}
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CTX[pid] = t[max(0, idx - 2500):idx]


def zero_out_hook(mod, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


for pid, ctx in CTX.items():
    ids = torch.tensor([tok(prompt_for(pid) + ctx).input_ids[-2000:]]).cuda()
    rows_out = {}
    for tag, hook_on in (('normal', False), ('L0-gdn-zero', True)):
        h = model.model.layers[0].linear_attn.register_forward_hook(zero_out_hook) if hook_on else None
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        if h:
            h.remove()
        lp = F.log_softmax(logits, -1)
        rows_out[tag] = float(lp[248069])
    print(f'  {pid}: P(</think>) normal {rows_out["normal"]:6.2f} | L0-gdn-zero {rows_out["L0-gdn-zero"]:6.2f}')
