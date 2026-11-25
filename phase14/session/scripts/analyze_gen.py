import json, re, sys, collections
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
n = len(rows)
print(f'{n} rows')
ok = sum(r['ok_boxed'] for r in rows)
print(f'accuracy_boxed {ok}/{n} = {100*ok/n:.1f}')
by = collections.defaultdict(lambda: [0,0])
for r in rows:
    p = practice[r['id']]
    by[p['level']][0] += r['ok_boxed']; by[p['level']][1] += 1
print('by level:', {k: f'{v[0]}/{v[1]}' for k,v in sorted(by.items())})
byt = collections.defaultdict(lambda: [0,0])
for r in rows:
    p = practice[r['id']]
    byt[p['type']][0] += r['ok_boxed']; byt[p['type']][1] += 1
print('by type:', {k: f'{v[0]}/{v[1]}' for k,v in sorted(byt.items())})
noclose = [r for r in rows if not r['closed']]
cap = [r for r in rows if r['finish']=='length']
nobox = [r for r in rows if r['pred_boxed'] is None and r['finish']!='length']
print(f'not closed: {len(noclose)}  cap-hit: {len(cap)}  no-boxed-but-finished: {len(nobox)}')
toks = sorted(r['n_tokens'] for r in rows)
print('token count: min', toks[0], 'p25', toks[n//4], 'median', toks[n//2], 'p75', toks[3*n//4], 'max', toks[-1])
print()
print('=== FAILED ids ===')
fails = [r['id'] for r in rows if not r['ok_boxed']]
print(' '.join(fails))
