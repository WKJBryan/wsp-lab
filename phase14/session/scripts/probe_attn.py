import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


FULL = [3, 7, 11, 15, 19, 23]

r = byid['p1']
p = prompt_for(r['id'])
text = r['text']
full_ids = tok(p + text).input_ids
ptok = tok(p).input_ids.__len__()
WIN = 3000
wids = torch.tensor([full_ids[-WIN:]]).to('cuda')
offset = len(full_ids) - WIN  # window starts here in full sequence
genpos = lambda i: i - ptok    # generation position of window index i

with torch.no_grad():
    out = model(wids, output_attentions=True)
atts = out.attentions
print('attention layers returned:', len(atts), 'shape', atts[0].shape)
nheads = atts[0].shape[1]

# decode window tokens for reference
wtoks = [tok.decode([t]) for t in wids[0].tolist()]

for name, wpos in [('loop-tail', WIN - 6), ('mid-window', WIN // 2), ('near-end-2', WIN - 130)]:
    gp = genpos(wpos)
    print(f'=== probe {name}: window pos {wpos} = gen pos {gp}, current token {wtoks[wpos]!r}')
    for li, A in enumerate(atts):
        layer = FULL[li]
        a = A[0, :, wpos].float()  # (heads, src in window)
        ent = -(a.clamp_min(1e-9).log() * a).sum(-1)
        last150 = a[:, -150:].sum(-1)
        last25 = a[:, -25:].sum(-1)
        front = a[:, :-150].sum(-1)
        bos = a[:, 0]
        top_src = a.argmax(-1)
        s = ' '.join(f'h{h}:{ent[h]:.2f}/L{last150[h]:.2f}/f{front[h]:.2f}@{top_src[h] - wpos}' for h in range(nheads))
        print(f'  L{layer:2d}: {s}')
print()
print('reference: loop period in tokens ~', 476 // 4)
for i in range(WIN - 12, WIN - 2):
    print(f'  {i}: {wtoks[i]!r}', end=' ')
print()
