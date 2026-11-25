import json, torch, sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()

for s in ['<think>', '</think>', '<|im_end|>', 'Wait']:
    print(repr(s), '->', tok.encode(s, add_special_tokens=False), [tok.decode([i]) for i in tok.encode(s, add_special_tokens=False)])

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
byid = {r['id']: r for r in rows}

def next_dist(prefix_text, k=10, targets=('</think>', 'Wait', '.', '\n')):
    ids = tok(prefix_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, k)
    tops = [(tok.decode([j]), round(float(x), 2)) for j, x in zip(i.tolist(), v.tolist())]
    tinfo = {}
    for t in targets:
        tid = tok.encode(t, add_special_tokens=False)
        if len(tid) == 1:
            tinfo[t] = (round(float(lp[tid[0]]), 2), int((lp > lp[tid[0]]).sum()))
    return tops, tinfo

# p0: loop unit 'Wait, ^{2018} = ...'
t = byid['p0']['text']
prompt = tok.apply_chat_template([{'role':'user','content':'Problem:\n' + practice['p0']['q'] + '\n\nPlease reason step by step, and put your final answer within \boxed{}.'}], add_generation_prompt=True, tokenize=False, enable_thinking=True)
# find first occurrence of the loop unit
unit = 'Wait, ^{2018} = x^{2015} \cdot x^3$.'
first = t.find(unit)
print('first loop unit at char', first, 'of', len(t))
for name, cut in [('before-any-Wait', t[:max(0, first-50)]), ('at-first-Wait', t[:first]), ('after-2-cycles', t[:first + 3*len(unit)]), ('deep-in-loop', t[-2000:])]:
    tops, ti = next_dist(prompt + cut)
    print(f'--- p0 {name}: top5={tops[:5]} </think>={ti.get("</think>")}')
