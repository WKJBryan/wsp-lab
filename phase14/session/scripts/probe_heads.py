import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}
FULL = [3, 7, 11, 15, 19, 23]


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


r = byid['p1']
p = prompt_for(r['id'])
ids = tok(p + r['text']).input_ids
WIN = 2500
wids = torch.tensor([ids[-WIN:]]).cuda()
NH, HD = 16, 128


def head_zero_hook(head, scale=0.0):
    def hook(mod, args):
        # out_proj input: (B, T, 2048); zero the head's 128 columns
        x = args[0]
        x[:, :, head * HD:(head + 1) * HD] = x[:, :, head * HD:(head + 1) * HD] * scale
        return (x,)
    return hook


@torch.no_grad()
def run_probe(head=None, scale=0.0, layer=0):
    handles = []
    if head is not None:
        mod = model.model.layers[layer].linear_attn.out_proj
        handles.append(mod.register_forward_pre_hook(head_zero_hook(head, scale)))
    logits = model(wids).logits[0, -1].float()
    for h in handles:
        h.remove()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, 4)
    return ' '.join(f'{tok.decode([j])!r}:{float(x):6.2f}' for j, x in zip(i.tolist(), v.tolist())), float(lp[248069])


s, th = run_probe()
print(f'base L0:   {s}  P(th)={th:.1f}')
for h in range(NH):
    s, th = run_probe(head=h)
    print(f'L0 h{h:2d} zero: {s}  P(th)={th:.1f}')
print()
# attenuation rather than full zeroing for the heads that matter will follow based on results
