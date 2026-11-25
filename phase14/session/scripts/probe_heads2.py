import json, torch, itertools
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/sandbox/model'
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


r = byid['p1']
p = prompt_for(r['id'])
ids = tok(p + r['text']).input_ids
WIN = 2500
wids = torch.tensor([ids[-WIN:]]).cuda()
NH, HD = 16, 128


def head_scale_hook(heads, scale):
    def hook(mod, args):
        x = args[0]
        for h in heads:
            x[:, :, h * HD:(h + 1) * HD] = x[:, :, h * HD:(h + 1) * HD] * scale
        return (x,)
    return hook


@torch.no_grad()
def run_probe(heads=(), scale=0.0, layer=0):
    handles = []
    if heads:
        mod = model.model.layers[layer].linear_attn.out_proj
        handles.append(mod.register_forward_pre_hook(head_scale_hook(heads, scale)))
    logits = model(wids).logits[0, -1].float()
    for h in handles:
        h.remove()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, 4)
    return ' '.join(f'{tok.decode([j])!r}:{float(x):6.2f}' for j, x in zip(i.tolist(), v.tolist())), float(lp[248069])


combos = {
    'fast heads (0,3,11,13)': [0, 3, 11, 13],
    'h3+h8': [3, 8],
    'slow heads (1,2,4..10,12,14,15)': [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15],
    'all except h3': [h for h in range(16) if h != 3],
    'all except h8': [h for h in range(16) if h != 8],
}
for name, heads in combos.items():
    s, th = run_probe(heads, 0.0)
    print(f'zero {name:34s}: {s}  P(th)={th:.1f}')

print()
for scale in (0.75, 0.5, 0.35, 0.25, 0.1):
    s, th = run_probe(list(range(16)), scale)
    print(f'scale ALL L0 heads x{scale:4.2f}          : {s}  P(th)={th:.1f}')
print()
for scale in (0.5, 0.25):
    s, th = run_probe([0, 3, 11, 13], scale)
    print(f'scale fast heads x{scale:4.2f}            : {s}  P(th)={th:.1f}')
    s, th = run_probe([8], scale)
    print(f'scale h8 x{scale:4.2f}                   : {s}  P(th)={th:.1f}')
    s, th = run_probe([3], scale)
    print(f'scale h3 x{scale:4.2f}                   : {s}  P(th)={th:.1f}')
