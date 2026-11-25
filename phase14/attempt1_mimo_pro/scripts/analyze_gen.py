import json, sys, re
from collections import Counter, defaultdict

gen_path = sys.argv[1]
rows = [json.loads(l) for l in open(gen_path, encoding='utf-8')]
prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
n = len(rows)
ok = [r for r in rows if r['ok_boxed']]
bad = [r for r in rows if not r['ok_boxed']]
print(f'n={n} ok_boxed={len(ok)} ({100*len(ok)/n:.1f}%) ok_rescored={sum(r["ok_rescored"] for r in rows)}')
print(f'closed={sum(r["closed"] for r in rows)} ({100*sum(r["closed"] for r in rows)/n:.1f}%)')
print(f'finish=length: {sum(r["finish"]=="length" for r in rows)}  mean_tokens={sum(r["n_tokens"] for r in rows)/n:.0f}')
# no boxed at all
nob = [r for r in rows if r['pred_boxed'] is None]
print(f'no \boxed at all: {len(nob)}')
# by level and type
for key in ('level', 'type'):
    d = defaultdict(lambda: [0,0])
    for r in rows:
        p = prac[r['id']]
        d[p[key]][0] += int(r['ok_boxed']); d[p[key]][1] += 1
    print(key, {k: f'{v[0]}/{v[1]}' for k, v in sorted(d.items())})
print()
print('=== FAILURES ===')
for r in bad:
    p = prac[r['id']]
    print(f"--- {r['id']} [{p['type']} L{p['level']}] gold={p['gold']!r} pred={r['pred_boxed']!r} rescore={r['pred_rescore']!r} toks={r['n_tokens']} closed={r['closed']} fin={r['finish']}")
