"""Find the actual thinking markers in the tokenizer vocab and in gen texts."""
import json
from transformers import AutoTokenizer

MD = '/home/user/wsp14/sandbox/model'
tok = AutoTokenizer.from_pretrained(MD)
for t, i in sorted(tok.get_vocab().items(), key=lambda x: x[1]):
    if 'think' in t.lower() or 'thinking' in t.lower():
        print(repr(t), i)

rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}
t66 = rows['p66']['text']
# hunt for any angle-bracketed marker near the end
import re
for m in re.finditer(r'<[^>]{1,30}>', t66):
    print('p66 marker:', repr(m.group()), 'at', m.start())
print('p66 tail:', repr(t66[-300:]))
