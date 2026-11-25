import json, re, sys, collections
rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_2.jsonl')]
practice = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json'))}
def find_loop_start(text, min_rep=3):
    # find the largest tail period p (in chars) with >= min_rep repetitions
    best = None
    n = len(text)
    for p in range(4, 400):
        unit = text[n-p:]
        if text[n-2*p:n-p] == unit and text[n-3*p:n-2*p] == unit:
            best = p
    return best
print('=== the 7 closed/finished generations ===')
for r in rows:
    if r['finish'] != 'length':
        print(r['id'], 'ntok', r['n_tokens'], 'closed', r['closed'], 'ok', r['ok_boxed'], 'gold', r['gold'], 'pred', r['pred_boxed'])
print()
print('=== loop stats for cap-hit generations ===')
starts = []
periods = []
for r in rows:
    if r['finish'] != 'length': continue
    t = r['text']
    p = find_loop_start(t)
    if p:
        # loop start position: earliest index i such that t[i:] is periodic with period p (approx: search backwards)
        n = len(t)
        unit = t[n-p:]
        i = n - p
        while i - p >= 0 and t[i-p:i] == unit:
            i -= p
        starts.append(i / n)
        periods.append(p)
    else:
        starts.append(1.0); periods.append(0)
import statistics
print(f'looping generations: {len(periods)}/{121}')
print(f'loop start position (fraction of text): median {statistics.median(starts):.2f}, mean {statistics.mean(starts):.2f}, min {min(starts):.2f}')
print(f'loop period (chars): median {statistics.median(periods)}, min {min(periods)}, max {max(periods)}')
# characterize loop units
units = collections.Counter()
for r in rows:
    if r['finish'] != 'length': continue
    t = r['text']
    p = find_loop_start(t)
    if p and p <= 120:
        units[t[-p:].replace(chr(10),' ')] += 1
for u, c in units.most_common(12):
    print(f'{c:3d}x  {u!r}')
