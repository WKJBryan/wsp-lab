# Probes A/B/C. Run only when no pass is active.
# A: teacher-force a correct-derivation prefix; is the right answer the top next token?
# C: P(</think>) trajectory along a generation (early close pressure).
import json, re, sys, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/user/wsp14/sandbox/model'
gen_path = sys.argv[2] if len(sys.argv) > 2 else '/home/user/wsp14/sandbox/harness/state/gen_1.jsonl'
n_probes = int(sys.argv[3]) if len(sys.argv) > 3 else 6

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()

practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
rows = [json.loads(l) for l in open(gen_path)]
fails = [r for r in rows if not r['ok_boxed']][:n_probes]

def top_next(text, k=5):
    ids = tok(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    v, i = torch.topk(lp, k)
    return [(tok.decode([j]), round(float(x), 2)) for j, x in zip(i.tolist(), v.tolist())]

print('================ PROBE A: correct derivation -> is the gold answer next? ================')
for r in fails:
    p = practice[r['id']]
    gold = str(p['gold'])
    msgs = [{'role':'user','content':'Problem:\n' + p['q'] + '\n\nPlease reason step by step, and put your final answer within \boxed{}.'}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    deriv = ('Let me work through this carefully.\n\nWe are given the problem. Working step by step, '
             'the computation gives the value.\n\nTherefore, the final answer is \boxed{')
    text = pre + deriv
    print(f"--- {r['id']} gold={gold!r} pred={str(r['pred_boxed'])[:20]!r} ---")
    print('   top next tokens after a (fake generic) derivation:', top_next(text))

print()
print('================ PROBE C: closing pressure along the generation ================')
for r in fails[:3]:
    ids_prompt = tok.apply_chat_template([{'role':'user','content':'Problem:\n' + practice[r['id']]['q'] + '\n\nPlease reason step by step, and put your final answer within \boxed{}.'}],
                                         add_generation_prompt=True, tokenize=False, enable_thinking=True)
    text = r['text']
    print(f"--- {r['id']} closed={r['closed']} ntok={r['n_tokens']} ---")
    for frac in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        cut = text[:int(len(text)*frac)]
        # find last newline-cut to keep tokens sane
        out = top_next(ids_prompt + cut, 8)
        think_lp = next((v for t, v in out if '</think>' in t or 'think' in t), None)
        print(f'   at {frac:.0%}: top8={out[:4]}')
