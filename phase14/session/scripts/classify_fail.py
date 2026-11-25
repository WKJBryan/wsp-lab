import json, re, sys, collections
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
fails = [r for r in rows if not r['ok_boxed']]
print(f'{len(fails)} failed of {len(rows)}')
cats = collections.Counter()
for r in fails:
    p = practice[r['id']]
    t = r['text']
    if r['finish'] == 'length': cats['cap-hit'] += 1; continue
    if not r['closed']: cats['never-closed-think'] += 1; continue
    if r['pred_boxed'] is None: cats['closed-but-no-boxed'] += 1; continue
    cats['wrong-answer'] += 1
print(dict(cats))
print()
print('=== detail: one line per failure ===')
for r in fails:
    p = practice[r['id']]
    t = r['text']
    waits = len(re.findall(r'(?i)\bwait\b|\bactually\b|let me re-?check|hold on', t))
    tail = t[-160:].replace(chr(10), ' | ')
    print(f"{r['id']} {p['type'][:12]:12s} {p['level'][-1]} ntok={r['n_tokens']:5d} closed={int(r['closed'])} fin={r['finish'][:4]:4s} waits={waits:2d} pred={str(r['pred_boxed'])[:30]:30s} gold={str(r['gold'])[:20]:20s}")
