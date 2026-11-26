"""Per-problem table of a generation file."""
import json, sys, re
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
import os; os.chdir('/home/user/wsp14/sandbox/harness'); sys.path.insert(0, '.')
from runcheck import SCRIPT
print('id   lvl type                   ntok  fin    ok pred_boxed | gold | #Wait #verify #boxed(in think) #nonlatin-tail')
for r in rows:
    p = prac[r['id']]
    t = r['text']
    think = t.split('</think>')[0] if '</think>' in t else t
    tail = t[-1500:]
    print('%-4s %s %-22s %6d %-6s %s %-28s | %-22s | %4d %4d %4d %4d' % (
        r['id'], p['level'][-1], p['type'][:22], r['n_tokens'], r['finish'], 'Y' if r['ok_boxed'] else '.',
        (str(r['pred_boxed'])[:28]), str(p['gold'])[:22],
        len(re.findall(r'\bWait\b', think)), len(re.findall(r'(?i)verif', think)), think.count('\boxed{'),
        len(SCRIPT.findall(tail))))
