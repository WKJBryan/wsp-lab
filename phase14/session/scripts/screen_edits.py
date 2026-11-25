"""Screen candidate layer-0 edits via out_proj column hooks on the test suite."""
import json, torch, sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/sandbox/model'
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


@torch.no_grad()
def run_suite(colscale=None):
    """colscale: list of (head, scale)."""
    handles = []
    if colscale:
        byhead = {}
        for h, s in colscale:
            byhead[h] = s

        def hook(mod, args):
            x = args[0]
            for h, s in byhead.items():
                x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * s
            return (x,)
        handles.append(model.model.layers[0].linear_attn.out_proj.register_forward_pre_hook(hook))
    res = {}
    for k in CTXIDS:
        logits = model(CTXIDS[k]).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 3)
        res[k] = ((tok.decode([i[0]]), float(v[0])), float(v[0] - v[1]), float(lp[248069]))
    for h in handles:
        h.remove()
    loops = [(t0, m, p) for kk, (t0, m, p) in res.items() if kk[0] == 'loop']
    closes = [(t0, m, p) for kk, (t0, m, p) in res.items() if kk[0] == 'close']
    return res, loops, closes


SLOW_NO_H8 = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15]
variants = {
    'base': None,
    'T4 all-but-h8 x0.3': [(h, 0.3) for h in range(16) if h != 8],
    'T8 all-but-h8 x0.1': [(h, 0.1) for h in range(16) if h != 8],
    'T8b all-but-h8 x0.05': [(h, 0.05) for h in range(16) if h != 8],
    'T9 all-but-h8 x0': [(h, 0.0) for h in range(16) if h != 8],
    'T9b keep h8x0.5': [(h, 0.0) for h in range(16) if h != 8] + [(8, 0.5)],
    'T10 keep h8+fast': [(h, 0.0) for h in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15)],
    'T11 keep h8 only x0.7': [(h, 0.0) for h in range(16) if h != 8] + [(8, 0.7)],
}
for name, cols in variants.items():
    res, loops, closes = run_suite(cols)
    lm = sum(m for _, m, _ in loops) / len(loops)
    cm = sum(p for _, _, p in closes) / len(closes)
    detail = ' '.join(f'{kk[1]}:{t0[0]!r}/{m:.1f}' for kk, (t0, m, p) in res.items() if kk[0] == 'loop')
    print(f'{name:24s} loop-margin-mean {lm:6.2f}  close-P(th)-mean {cm:6.2f}')
    print(f'    {detail}')
