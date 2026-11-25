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
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \boxed{}.'
    return tok.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
pid = 'p0'
ids = tok(prompt_for(pid) + byid[pid]['text']).input_ids[:6000]
x = torch.tensor([ids]).cuda()
def zero_hook(mod, inp, out):
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)
FULL = [3, 7, 11, 15, 19, 23]
probe_positions = [300, 2500, 5990]
for tag, zero in (('normal', False), ('L0-gdn-zero', True)):
    h = model.model.layers[0].linear_attn.register_forward_hook(zero_hook) if zero else None
    with torch.no_grad():
        out = model(x, output_attentions=True)
    if h: h.remove()
    print('===', tag)
    for li, A in enumerate(out.attentions):
        L = FULL[li]
        row = []
        for hh_ in range(8):
            cells = []
            for p in probe_positions:
                a = A[0, hh_, p].float()
                cells.append('%.3f' % float(a[0]))
            row.append('h%d:%s' % (hh_, '/'.join(cells)))
        print(' L%d sink-mass@{300/2500/5990}: ' % L + '  '.join(row))
