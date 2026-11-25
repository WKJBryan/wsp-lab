"""Per-channel map of h8 REMAINING channels on gen_2 loop contexts (cand_C weights).
Score: loop margins (10 problems), close P(th), sanity prompts."""
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


def ch_hook(channels, scale=0.0):
    def hook(mod, args):
        x = args[0]
        for c in channels:
            x[:, :, H8 + c] = x[:, :, H8 + c] * scale
        return (x,)
    return hook


@torch.no_grad()
def measure(channels=None):
    hh = OP.register_forward_pre_hook(ch_hook(channels)) if channels else None
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


base_lm, base_cl, base_san = measure()
print('PIDS:', PIDS)
print('base loop: ' + ' '.join(f'{k}:{v:.1f}' for k, v in base_lm.items()))
print('base close:', {k: round(v, 1) for k, v in base_cl.items()})
print('base sanity:', base_san)
print()
res = []
for c in range(HD):
    lm, cl, san = measure([c])
    dl = sum(lm[p] - base_lm[p] for p in PIDS) / len(PIDS)
    dmax = max(lm[p] - base_lm[p] for p in PIDS)
    dc = sum(cl.values()) / len(cl) - sum(base_cl.values()) / len(base_cl)
    broke = (san[0][:3] != base_san[0][:3]) or (san[1][:4] != base_san[1][:4])
    if dl < -0.5 or dmax < -3:
        res.append((c, dl, dmax, dc, broke))
        print(f'  h8ch{c:3d}: dLoopMean {dl:+6.2f} dLoopMax {dmax:+6.2f} dClose {dc:+6.2f} broke={broke}')
print()
print('candidate channels:', [(c, round(a, 2)) for c, a, b, cc, d in res])
