"""Compare the FINAL-ANSWER phase (after the first </think>) of generations that ended with a
boxed answer against those that ran to the cap.  Usage: python fpmix.py <gen.jsonl>"""
import sys, json, re
path = sys.argv[1]
G = [json.loads(l) for l in open(path, encoding='utf-8')]
BOX = chr(92) + 'boxed'
good, bad = [], []
for g in G:
    t = g['text']
    if '</think>' not in t:
        continue
    th, post = t.split('</think>', 1)
    rec = dict(id=g['id'], ntok=g['n_tokens'], ok=g['ok_boxed'], think=len(th), post=len(post), waits=post.count('Wait'),
               closers=t.count('</think>'), boxes=post.count(BOX), last_box=None, first_box=None)
    k = post.rfind(BOX)
    if k >= 0:
        rec['last_box'] = post[max(0, k - 160):k + 120]
        rec['after_last_box'] = post[k + 5:][-200:] if len(post) - k > 5 else ''
    (good if g['finish'] == 'stop' else bad).append(rec)
for name, group in [('ENDED (stop)', good), ('CAPPED (length)', bad)]:
    n = len(group)
    print(f'\n== {name}: {n} generations with a final phase')
    if n:
        med = lambda key: sorted(r[key] for r in group)[n // 2]
        print(f'   median thinking chars {med("think")}, final-phase chars {med("post")}, "Wait" in final phase {med("waits")} (mean {sum(r["waits"] for r in group)/n:.0f}), '
              f'median </think> count {med("closers")}, share with >1 </think>: {sum(1 for r in group if r["closers"] > 1)/n:.2f}, correct: {sum(1 for r in group if r["ok"])}/{n}')
print('\n== how the last boxed answer looks in 8 ended generations (160 chars before, 120 after):')
for r in good[::max(1, len(good) // 8)][:8]:
    print(f'  {r["id"]} (final phase {r["post"]} chars, {r["waits"]} Waits, {r["closers"]} </think>): ...{r["last_box"]!r}')
print('\n== ended generations with the longest final phases:')
for r in sorted(good, key=lambda r: -r['post'])[:4]:
    print(f'  {r["id"]}: final phase {r["post"]} chars, {r["waits"]} Waits, {r["closers"]} </think>, ok={r["ok"]}: ...{r["last_box"]!r}')
