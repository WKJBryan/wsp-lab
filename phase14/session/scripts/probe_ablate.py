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


def zero_out_hook(mod, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


@torch.no_grad()
def run_with_zeroed(attn_layers, mlp_layers):
    handles = []
    for L in attn_layers:
        mod = model.model.layers[L].self_attn if L in FULL else model.model.layers[L].linear_attn
        handles.append(mod.register_forward_hook(zero_out_hook))
    for L in mlp_layers:
        handles.append(model.model.layers[L].mlp.register_forward_hook(zero_out_hook))
    logits = model(wids).logits[0, -1].float()
    for h in handles:
        h.remove()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, 5)
    return [(tok.decode([j]), round(float(x), 2)) for j, x in zip(i.tolist(), v.tolist())], float(lp[248069])


base, th = run_with_zeroed([], [])
print('no ablation:            ', base, ' P(</think>)=', round(th, 1))
for L in FULL:
    ab, th2 = run_with_zeroed([L], [])
    print(f'zero attn L{L:2d}:          ', ab[:3], ' P(</think>)=', round(th2, 1))
for L in [L for L in range(24) if L not in FULL][:9]:
    ab, th2 = run_with_zeroed([L], [])
    print(f'zero gdn  L{L:2d}:          ', ab[:3], ' P(</think>)=', round(th2, 1))
for L in range(24):
    ab, th2 = run_with_zeroed([], [L])
    print(f'zero mlp  L{L:2d}:          ', ab[:3], ' P(</think>)=', round(th2, 1))
ab, th2 = run_with_zeroed(FULL, [])
print('zero ALL attn:          ', ab[:3], ' P(</think>)=', round(th2, 1))
ab, th2 = run_with_zeroed([L for L in range(24) if L not in FULL], [])
print('zero ALL gdn:           ', ab[:3], ' P(</think>)=', round(th2, 1))
