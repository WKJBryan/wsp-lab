"""Per-head map of L1 and L2 GDN out_proj on gen_2 loop contexts (cand_C weights)."""
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


def head_hook(layer, head, scale):
    OP = model.model.layers[layer].linear_attn.out_proj

    def hook(mod, args):
        x = args[0]
        x[:, :, head * HD:(head + 1) * HD] = x[:, :, head * HD:(head + 1) * HD] * scale
        return (x,)
    return hook


@torch.no_grad()
def measure(layer=None, head=None):
    hh = None
    if layer is not None:
        hh = model.model.layers[layer].linear_attn.out_proj.register_forward_pre_hook(head_hook(layer, head, 0.0))
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
    if hh:
        hh.remove()
    return lm, cl, san


base_lm, base_cl, base_san = measure()
blm = sum(base_lm.values()) / len(base_lm)
bcl = sum(base_cl.values()) / len(base_cl)
print('base loop mean %.2f close %.2f san %r' % (blm, bcl, base_san[0][:12]))
for L in (1, 2):
    for h in range(16):
        lm, cl, san = measure(L, h)
        dl = sum(lm.values()) / len(lm) - blm
        dc = sum(cl.values()) / len(cl) - bcl
        broke = san[0][:3] != base_san[0][:3]
        if dl < -0.4:
            print(f'  L{L} h{h:2d}: dLoop {dl:+6.2f} [{ " ".join(f"{k}:{lm[k]-base_lm[k]:+.1f}" for k in PIDS if lm[k]-base_lm[k] < -0.5) }] dClose {dc:+5.2f} broke={broke}')
