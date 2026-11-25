"""Chat-mode NLL of real successful generations: cand_C vs +attn x0.35/x0.5."""
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
FULL = [3, 7, 11, 15, 19, 23]
E = model.get_input_embeddings().weight


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


def scale_hook(c):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] * c,) + out[1:]
        return out * c
    return hook


@torch.no_grad()
def nll_of(pid, c_attn=None):
    t = byid[pid]['text']
    ids = torch.tensor([tok(prompt_for(pid) + t).input_ids[-3500:]]).cuda()
    handles = [model.model.layers[L].self_attn.register_forward_hook(scale_hook(c_attn)) for L in FULL] if c_attn else []
    logits = model(ids).logits[0].float()
    for h in handles:
        h.remove()
    lp = F.log_softmax(logits[:-1], -1)
    tgt = ids[0, 1:]
    toknll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(toknll.mean()), toknll


PIDS = ['p1', 'p2', 'p16', 'p95', 'p112', 'p28', 'p105']
for c in (None, 0.5, 0.35):
    vals = []
    per = {}
    for pid in PIDS:
        m, tk = nll_of(pid, c)
        vals.append(m)
        per[pid] = round(m, 3)
    print(f'attn x{c}: mean NLL {sum(vals) / len(vals):.3f}  per-gen {per}')
