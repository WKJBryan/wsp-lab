"""Teacher-forced next-token distribution at chosen offsets of a generation.
Usage: next_tok.py <gen_id> <char_offset> [<char_offset>...]
Prints top-20 next tokens plus the rank/logit of selected tokens of interest (GPU bf16)."""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_ID = sys.argv[1]
OFFSETS = [int(x) for x in sys.argv[2:]]
MD = '/home/user/wsp14/sandbox/model'
REF_Q = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{}."

tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16, device_map='cuda')
model.eval()

prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}
p = prac[GEN_ID]
text = rows[GEN_ID]['text']

prompt = tok.apply_chat_template([{'role': 'user', 'content': REF_Q.replace('{q}', p['q'])}],
                                 add_generation_prompt=True, tokenize=False, enable_thinking=True)
prompt_ids = tok(prompt, return_tensors='pt').input_ids[0]
prompt_len = prompt_ids.shape[0]

INTEREST = {
    'think_end(248069)': 248069,
    'think_start(248068)': 248068,
    'Wait_token': tok.convert_tokens_to_ids('Wait') if 'Wait' in tok.get_vocab() else None,
    'ĊĊ5': None,
}
# find ids for some literal prefixes of interest
for lit in [' Wait', 'Wait', '\n\n', ' Let', ' But', ' The']:
    tid = tok(lit, add_special_tokens=False).input_ids
    if len(tid) == 1:
        INTEREST[lit] = tid[0]

for off in OFFSETS:
    prefix = text[:off]
    gen_ids = tok(prefix, add_special_tokens=False, return_tensors='pt').input_ids[0]
    ids = torch.cat([prompt_ids, gen_ids]).unsqueeze(0).to('cuda')
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 20)
    print(f'== {GEN_ID} offset={off} (tokens so far: {prompt_len}+{gen_ids.shape[0]}) ==')
    print('context tail:', repr(prefix[-120:]))
    for pv, pi in zip(top.values, top.indices):
        print(f'   {float(pv):.4f}  {int(pi):6d}  {tok.decode([int(pi)])!r}')
    print('  -- tokens of interest --')
    for name, tid in INTEREST.items():
        if tid is None:
            continue
        rank = int((logits > logits[tid]).sum()) + 1
        print(f'   {name}: logit={float(logits[tid]):.3f} prob={float(probs[tid]):.5f} rank={rank}')
    print()
