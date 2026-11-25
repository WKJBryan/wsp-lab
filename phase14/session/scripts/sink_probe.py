"""Attention-sink probe: per-head mass on position 0 (BOS) and entropy vs context position."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/cand_C'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
model.eval()

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_2.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=True)


# use a cap-hit generation truncated to ~6000 tokens (fits eager attention)
pid = 'p0'
ids = tok(prompt_for(pid) + byid[pid]['text']).input_ids[:6000]
x = torch.tensor([ids]).cuda()
with torch.no_grad():
    out = model(x, output_attentions=True)
FULL = [3, 7, 11, 15, 19, 23]
n = x.shape[1]
probe_positions = [50, 300, 1000, 2500, 5000, 5990]
print('context length', n)
for li, A in enumerate(out.attentions):
    L = FULL[li]
    print(f'--- L{L} (8 heads): mass on position 0 | entropy, at positions {probe_positions}')
    for h in range(8):
        cells = []
        for p in probe_positions:
            a = A[0, h, p].float()
            sink = float(a[0])
            ent = float(-(a.clamp_min(1e-9).log() * a).sum())
            cells.append(f'{sink:.3f}/{ent:.1f}')
        print(f'  h{h}: ' + '  '.join(cells))
