"""Characterize where generations enter the repetition loop.
For each row: find first position where any 60-char window's content has already occurred
(a repeated n-gram), report onset position, onset context, and the repeated phrase."""
import json, sys, re
from collections import Counter

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding='utf-8')]

def loop_info(text, gram=200):
    """Return (onset_idx, phrase) of first exact repeated gram (by chars), else (None, None)."""
    seen = {}
    for i in range(0, len(text) - gram, 20):
        g = text[i:i + gram]
        if g in seen:
            return seen[g], g
        seen[g] = i
    return None, None

n_loop = 0
for r in rows:
    t = r['text']
    onset, phrase = loop_info(t)
    if onset is None:
        status = 'NO-LOOP'
    else:
        status = f'LOOP@{onset}/{len(t)}'
        n_loop += 1
    print(f"{r['id']}: toks={r['n_tokens']} closed={r['closed']} ok={r['ok_boxed']} {status}")
    if onset is not None:
        ctx = t[max(0, onset - 250):onset + 50].replace('\n', ' | ')
        print(f"   onset ctx: ...{ctx}")
        print(f"   phrase: {phrase[:120]!r}")
print(f'\n{n_loop}/{len(rows)} generations contain an exact repeated 200-char gram')
