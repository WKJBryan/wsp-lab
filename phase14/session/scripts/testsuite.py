"""Test suite: candidate edits vs contexts (loop breaking + closing preservation)."""
import json, torch, sys, importlib
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/user/wsp14/work/cand_A'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}
HD = 128
NH = 16


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


def find_loop(text):
    n = len(text)
    for cand in range(4, 500):
        unit = text[n - cand:]
        if text[n - 2 * cand:n - cand] == unit and text[n - 3 * cand:n - 2 * cand] == unit:
            period = cand
    try:
        period
    except UnboundLocalError:
        return None
    unit = text[n - period:]
    start = n - period
    while start - period >= 0 and text[start - period:start] == unit:
        start -= period
    return start, unit


def contexts():
    out = {}
    for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40']:
        t = byid[pid]['text']
        fl = find_loop(t)
        if not fl:
            continue
        start, unit = fl
        out[('loop', pid)] = byid[pid]['text'][start:start + 3 * len(unit)]
    for pid in ['p124', 'p28', 'p101', 'p27']:
        t = byid[pid]['text']
        idx = t.find('</think>')
        out[('close', pid)] = t[max(0, idx - 2500):idx]
    for pid in ['p28', 'p101']:
        t = byid[pid]['text']
        out[('mid', pid)] = t[:int(0.5 * len(t))]
    return out


CTX = contexts()
CTXIDS = {}
for k in CTX:
    ids = tok(prompt_for(k[1]) + CTX[k]).input_ids
    CTXIDS[k] = torch.tensor([ids[-2500:]]).cuda()
print(f'{len(CTX)} contexts loaded')


@torch.no_grad()
def eval_ctx(key):
    logits = model(CTXIDS[key]).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, 3)
    tops = [(tok.decode([j]), round(float(x), 2)) for j, x in zip(i.tolist(), v.tolist())]
    margin = float(v[0] - v[1])
    return tops, margin, float(lp[248069])


def report(tag):
    print(f'===== {tag} =====')
    agg_loop, agg_close = [], []
    for k in CTX:
        tops, margin, pth = eval_ctx(k)
        kind, pid = k
        if kind == 'loop':
            agg_loop.append(margin)
            print(f'  loop {pid}: top {tops[0][0]!r} margin {margin:6.2f} (2nd {tops[1][0]!r} {tops[1][1]:5.2f})  P(th) {pth:6.1f}')
        elif kind == 'close':
            agg_close.append(pth)
            print(f'  close {pid}: P(</think>) {pth:6.2f}  top {tops[0][0]!r} {tops[0][1]:5.2f}')
        else:
            print(f'  mid {pid}: top {tops[0][0]!r} {tops[0][1]:5.2f} margin {margin:5.2f}  P(th) {pth:6.1f}')
    if agg_loop:
        print(f'  >> loop margins: mean {sum(agg_loop) / len(agg_loop):.2f} (want < ~1.5)')
    if agg_close:
        print(f'  >> close P(</think>): mean {sum(agg_close) / len(agg_close):.2f} (want <= ~3)')


report('BASE (current weights)')
