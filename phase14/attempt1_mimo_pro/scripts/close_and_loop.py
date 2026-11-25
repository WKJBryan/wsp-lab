"""(1) Next-token distribution at p66's healthy close decision (offset 6077).
(2) Loop inducibility in a clean context: teacher-forced next-token after a phrase repeated N times."""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = '/home/user/wsp14/sandbox/model'
TAG = '<' + '|end' + 'of' + 'thinking|' + '>'
REF_Q = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map='cuda')
model.eval()

prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}


def show_top(ids, label, interests):
    with torch.no_grad():
        logits = model(ids.to('cuda')).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 15)
    print(f'== {label} ==')
    for pv, pi in zip(top.values, top.indices):
        print(f'   {float(pv):.4f}  {int(pi):6d}  {tok.decode([int(pi)])!r}')
    for name, tid in interests.items():
        rank = int((logits > logits[tid]).sum()) + 1
        print(f'   [{name}] logit={float(logits[tid]):.3f} prob={float(probs[tid]):.6f} rank={rank}')
    print()


INTEREST = {'think_end': 248069, 'think_start': 248068}
for lit in [' Wait', ' Okay', ' Looks', ' The', ' So', ' Let', ' Now', ' I', '\n\n', ' Correct', ' Hmm']:
    t = tok(lit, add_special_tokens=False).input_ids
    if len(t) == 1:
        INTEREST[lit] = t[0]

# (1) p66 healthy close
p66 = prac['p66']
t66 = rows['p66']['text']
prompt66 = tok.apply_chat_template([{'role': 'user', 'content': REF_Q.replace('{q}', p66['q'])}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=True)
pid66 = tok(prompt66, return_tensors='pt').input_ids[0]
for off in (6077, 6060):
    gid = tok(t66[:off], add_special_tokens=False, return_tensors='pt').input_ids[0]
    ids = torch.cat([pid66, gid]).unsqueeze(0)
    print(f'p66 context tail: {t66[max(0, off-100):off]!r}')
    show_top(ids, f'p66 close decision offset={off}', INTEREST)

# (2) loop inducibility in clean context
p2 = prac['p2']
prompt2 = tok.apply_chat_template([{'role': 'user', 'content': REF_Q.replace('{q}', p2['q'])}],
                                  add_generation_prompt=True, tokenize=False, enable_thinking=True)
pid2 = tok(prompt2, return_tensors='pt').input_ids[0]
phrase = "Wait, is it possible that the user meant $\\lfloor0.999\\rfloor + \\lceil2.001\\rceil$?\n    *   "
for nrep in (1, 3, 8):
    # clean context: short legit thinking then the phrase repeated nrep times
    lead = "Let me solve this step by step.\n    *   $\\lfloor0.999\\rfloor = 0$.\n    *   $\\lceil2.001\\rceil = 3$.\n    *   Sum = 3.\n    *   "
    body = lead + phrase * nrep
    gid = tok(body, add_special_tokens=False, return_tensors='pt').input_ids[0]
    ids = torch.cat([pid2, gid]).unsqueeze(0)
    show_top(ids, f'CLEAN loop test nrep={nrep}', INTEREST)
