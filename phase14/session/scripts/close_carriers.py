"""Which branches carry closing? Per-layer ablation on close contexts."""
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
FULL = [3, 7, 11, 15, 19, 23]
GDN = [L for L in range(24) if L not in FULL]


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


CTX = {}
for pid in ['p124', 'p28', 'p101', 'p27']:
    t = byid[pid]['text']
    idx = t.find('</think>')
    CTX[pid] = torch.tensor([tok(prompt_for(pid) + t[max(0, idx - 2500):idx]).input_ids[-2000:]]).cuda()


def zero_hook(mod, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


@torch.no_grad()
def pth(zero_attn=None, zero_mlp=None):
    handles = []
    for L in (zero_attn or []):
        mod = model.model.layers[L]
        handles.append((mod.self_attn if L in FULL else mod.linear_attn).register_forward_hook(zero_hook))
    for L in (zero_mlp or []):
        handles.append(model.model.layers[L].mlp.register_forward_hook(zero_hook))
    out = []
    for k, ids in CTX.items():
        logits = model(ids).logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        v, i = torch.topk(lp, 2)
        out.append((k, float(lp[248069]), tok.decode([i[0]])))
    for h in handles:
        h.remove()
    return out


base = pth()
print('base:', [(k, round(p, 1)) for k, p, t in base])
for L in range(24):
    a = pth(zero_attn=[L])
    m = pth(zero_mlp=[L])
    pa = sum(p for _, p, _ in a) / len(a)
    pm = sum(p for _, p, _ in m) / len(m)
    flag = ' <<<' if pa > -4 or pm > -4 else ''
    print(f'zero L{L:2d} attn: {pa:6.2f} | zero L{L:2d} mlp: {pm:6.2f}{flag}')
