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


def prompt_for(pid):
    q = practice[pid]['q']
    content = 'Problem:\n' + q + '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    msgs = [{'role': 'user', 'content': content}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)


def probe(prefix_text, note='', k=6):
    ids = tok(prefix_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, k)
    tops = [(tok.decode([j]).replace('\n', '|n'), round(float(x), 2)) for j, x in zip(i.tolist(), v.tolist())]
    tid = 248069  # </think>
    lt = float(lp[tid])
    rank = int((lp > lt).sum())
    print(f'  {note:24s} P(</think>)={lt:7.2f} (rank {rank:6d})  top{k}: {tops}')


for pid in ['p0', 'p1', 'p2', 'p5']:
    r = byid[pid]
    t = r['text']
    p = prompt_for(pid)
    n = len(t)
    period = None
    for cand in range(4, 500):
        unit = t[n - cand:]
        if t[n - 2 * cand:n - cand] == unit and t[n - 3 * cand:n - 2 * cand] == unit:
            period = cand
    if not period:
        print(pid, 'no simple period')
        continue
    unit = t[n - period:]
    start = n - period
    while start - period >= 0 and t[start - period:start] == unit:
        start -= period
    print(f'=== {pid} loop started at char {start}/{n} ({start / n:.0%}), period {period}, unit={unit[:70]!r}')
    probe(p + t[:start], 'just-before-loop')
    probe(p + t[:start + 2 * period], '2-cycles-in')
    probe(p + t[-1500:], 'deep-in-loop')
    probe(p + t[:int(n * 0.15)], 'at-15%-of-text')
    print()

for pid in ['p124', 'p28']:
    r = byid[pid]
    t = r['text']
    p = prompt_for(pid)
    n = len(t)
    print(f'=== {pid} SUCCESS closed={r["closed"]} ntok={r["n_tokens"]}')
    for frac in (0.3, 0.5, 0.7, 0.85, 0.95, 1.0):
        probe(p + t[:int(n * frac)], f'at {frac:.0%}')
