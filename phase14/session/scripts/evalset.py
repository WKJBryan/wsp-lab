"""Combined evaluator: loop margins (12 problems), close P(th), sanity prompts.
Used to score channel-level candidates without spending passes."""
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = sys_arg = '/home/user/wsp14/work/model_pristine'
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


LOOP, CLOSE = {}, {}
for pid in ['p1', 'p0', 'p2', 'p5', 'p9', 'p40', 'p3', 'p10', 'p20', 'p30', 'p50', 'p70']:
    t = byid[pid]['text']
    fl = find_loop(t)
    if fl:
        LOOP[pid] = torch.tensor([tok(prompt_for(pid) + byid[pid]['text'][fl[0]:fl[0] + 3 * len(fl[1])]).input_ids[-2500:]]).cuda()
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CLOSE[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()

SANity_PROMPTS = ['2 + 2 =', 'Q: What is 17 * 24? A:', 'The capital of France is']
SANITY_IDS = [tok(p, return_tensors='pt').input_ids.cuda() for p in SANity_PROMPTS]

OP = model.model.layers[0].linear_attn.out_proj


def spec_hook(channels, scale=0.0):
    def hook(mod, args):
        x = args[0]
        for c in channels:
            x[:, :, c] = x[:, :, c] * scale
        return (x,)
    return hook


@torch.no_grad()
def evaluate(channels):
    hh = OP.register_forward_pre_hook(spec_hook(channels)) if channels else None
    lm = {}
    for pid, ids in LOOP.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        lm[pid] = float(v[0] - v[1])
    cl = []
    for pid, ids in CLOSE.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        cl.append(float(lp[248069]))
    san = []
    for ids in SANITY_IDS:
        out = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        san.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    if hh:
        hh.remove()
    return lm, cl, san


if __name__ == '__main__':
    base_lm, base_cl, base_san = evaluate([])
    print('base loop:', {k: round(v, 1) for k, v in base_lm.items()})
    print('base close:', [round(v, 1) for v in base_cl], 'sanity:', base_san)
    print()
    print('h3 channel map (loop-mean delta, loop-max delta, close delta):')
    res = []
    for c in range(HD):
        lm, cl, san = evaluate([3 * HD + c])
        dmean = sum(lm.values()) / len(lm) - sum(base_lm.values()) / len(base_lm)
        dmax = max(lm.values()) - max(base_lm.values())
        dcl = sum(cl) / len(cl) - sum(base_cl) / len(base_cl)
        res.append((c, dmean, dmax, dcl, san))
    for c, dmean, dmax, dcl, san in res:
        flag = ''
        if dmean < -0.8:
            flag += ' LOOP'
        if dcl > 0.8 or '2' == san[0].strip().lstrip()[0:1] and '2 + 2' in SANity_PROMPTS[0]:
            pass
        print(f'  h3ch{c:3d}: dLoopMean {dmean:+6.2f} dLoopMax {dmax:+6.2f} dClose {dcl:+6.2f}  san0={san[0][:14]!r} san1={san[1][:24]!r}{flag}')
