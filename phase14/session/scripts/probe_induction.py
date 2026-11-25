"""Measure full-attn head attention mass at distance ~= loop period at loop positions."""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = '/home/user/wsp14/work/model_try1'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
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


pid = 'p1'
t = byid[pid]['text']
n = len(t)
period = None
for cand in range(4, 500):
    unit = t[n - cand:]
    if t[n - 2 * cand:n - cand] == unit and t[n - 3 * cand:n - 2 * cand] == unit:
        period = cand
unit = t[n - period:]
start = n - period
while start - period >= 0 and t[start - period:start] == unit:
    start -= period
print(f'period {period} chars')
full_ids = tok(prompt_for(pid) + t).input_ids
P_tok = len(tok(unit).input_ids)
print(f'period {P_tok} tokens; loop starts at gen token {len(tok(t[:start]).input_ids)}')

WIN = 2400
wids = torch.tensor([full_ids[-WIN:]]).cuda()
wlen = len(full_ids) - WIN

with torch.no_grad():
    out = model(wids, output_attentions=True)

# loop positions inside window: the loop occupies the tail. Distance back to previous cycle = P_tok.
# sample positions every 100 within the last 1000 window positions
print('attention mass at distance P (previous cycle same position), per head. Also distance 2P.')
print('layer | ' + ' | '.join(f'h{h}' for h in range(8)))
for li, A in enumerate(atts := out.attentions):
    layer = FULL[li]
    masses = torch.zeros(8, device='cuda')
    masses2 = torch.zeros(8, device='cuda')
    cnt = 0
    for wpos in range(WIN - 1200, WIN - 40, 100):
        a = A[0, :, wpos].float()  # (8, src)
        for h in range(8):
            lo, hi = max(0, wpos - P_tok - 2), wpos - P_tok + 3
            masses[h] += a[h, lo:hi].sum()
            lo2 = max(0, wpos - 2 * P_tok - 2)
            masses2[h] += a[h, lo2:wpos - 2 * P_tok + 3].sum()
        cnt += 1
    print(f'L{layer:2d} P  | ' + ' | '.join(f'{m / cnt:.3f}' for m in masses))
    print(f'L{layer:2d} 2P | ' + ' | '.join(f'{m / cnt:.3f}' for m in masses2))
