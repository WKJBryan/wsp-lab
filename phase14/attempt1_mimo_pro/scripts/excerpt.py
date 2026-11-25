"""Print excerpts of generations: head, middle, tail. Usage: excerpt.py gen.jsonl idx [idx...]
With no idx, uses 0 1 2."""
import json, sys

path = sys.argv[1]
idxs = [int(x) for x in sys.argv[2:]] or [0, 1, 2]
rows = [json.loads(l) for l in open(path, encoding='utf-8')]
for i in idxs:
    r = rows[i]
    t = r['text']
    print(f'===== {r["id"]} toks={r["n_tokens"]} closed={r["closed"]} ok={r["ok_boxed"]} =====')
    n = len(t)
    print('--- HEAD (0:1200) ---')
    print(t[:1200])
    print(f'--- MID ({n//2}:{n//2+800}) ---')
    print(t[n // 2:n // 2 + 800])
    print(f'--- TAIL (-1200:) ---')
    print(t[-1200:])
    print()
