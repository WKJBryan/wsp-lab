"""Tokenize specials + build the eval's exact prompts; find decision-point offsets in gen texts."""
import json
from transformers import AutoTokenizer

MD = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(MD)

print('== special tokens ==')
for name, tid in sorted(tok.get_vocab().items()):
    if tok.convert_ids_to_tokens(tid) != name or any(ch in name for ch in '<>|'):
        pass
print('all special:', tok.all_special_tokens, tok.all_special_ids)
extra = [t for t in tok.get_vocab() if t.startswith('<') or t.endswith('>')]
print('angle tokens:', extra[:40])
for t in extra:
    print(repr(t), tok.convert_tokens_to_ids(t))

REF_Q = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{}."
prac = json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))
p2 = next(p for p in prac if p['id'] == 'p2')
p66 = next(p for p in prac if p['id'] == 'p66')

prompt2 = tok.apply_chat_template([{'role': 'user', 'content': REF_Q.replace('{q}', p2['q'])}],
                                  add_generation_prompt=True, tokenize=False, enable_thinking=True)
print('== prompt for p2 ==')
print(repr(prompt2[:600]))
print('prompt2 token len:', len(tok(prompt2).input_ids))

rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}
t2 = rows['p2']['text']
t66 = rows['p66']['text']
# decision points
idx = t2.find('Expression: $0 + 3 = 3$.')
print('p2 decision offset:', idx, repr(t2[idx:idx + 80]))
idx66 = t66.find('Looks good.')
print('p66 decision offset:', idx66, repr(t66[idx66:idx66 + 80]))
# loop point: after many repeats
lp = t2.find('Wait, is it possible that the user meant')
print('p2 first doubt offset:', lp, repr(t2[lp - 60:lp + 80]))
n_rep = t2.count('Wait, is it possible that the user meant')
print('p2 loop phrase count:', n_rep)
lp2 = t2.rfind('Wait, is it possible that the user meant')
print('p2 last loop offset:', lp2)
