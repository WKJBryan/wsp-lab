"""Verify the numbers quoted in the final report. CPU only."""
import json
import sys
from collections import Counter

sys.path.insert(0, '/home/user/wsp14/work')
import reads as R

meta = {p['id']: p for p in R.K.PRACTICE}
S = '/home/user/wsp14/sandbox/harness/state/'
ok = True


def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(('OK   ' if good else 'DIFF ') + name + ': ' + str(got) + ('' if good else f'  (report says {want})'))


def summary(n):
    return json.load(open(S + f'eval_pass_{n}.json'))


# per-attempt headline numbers
want = {1: (7, 7, 121, 31286), 2: (40, 42, 87, 25995), 3: (48, 58, 79, 22700), 4: (44, 54, 80, 22584), 5: (46, 84, 71, 21351),
        7: (48, 58, 79, 22700), 8: (23, 22, 107, 29376), 9: (44, 55, 80, 22743), 10: (46, 57, 77, 21794), 11: (30, 33, 96, 27867),
        12: (33, 39, 95, 25295)}
for n, (r, c, cap, mt) in want.items():
    rows = R.gen_rows(n)
    s = summary(n)
    check(f'attempt {n}: right', round(128 * s['accuracy_boxed'] / 100), r)
    check(f'attempt {n}: ended', sum(x['closed'] for x in rows.values()), c)
    check(f'attempt {n}: cut at cap', sum(x['finish'] == 'length' for x in rows.values()), cap)
    check(f'attempt {n}: mean tokens', round(s['mean_tokens'] / 10) * 10 if False else s['mean_tokens'], mt)
    check(f'attempt {n}: failed checks', s['failed_checks'], [])
# leak counts
for n, lk in {3: 9, 9: 7, 10: 6, 4: 6, 5: 27, 12: 6}.items():
    rows = R.gen_rows(n)
    check(f'attempt {n}: ended but cut at cap (leak)', sum(x['closed'] and x['finish'] != 'stop' for x in rows.values()), lk)
# by level, summed over the final weights' three copies
lv = Counter(); tot = Counter()
for n in (3, 9, 10):
    for pid, x in R.gen_rows(n).items():
        lv[meta[pid]['level']] += int(x['ok_boxed']); tot[meta[pid]['level']] += 1
print('final weights by level:', {k: f'{lv[k]} of {tot[k]}' for k in sorted(lv)}, 'sum', sum(lv.values()), 'of', sum(tot.values()),
      f'= {100 * sum(lv.values()) / sum(tot.values()):.1f}%')
lv2 = Counter(); tot2 = Counter(); nc = Counter(); nct = Counter()
for n in (2, 8, 11):
    for pid, x in R.gen_rows(n).items():
        lv2[meta[pid]['level']] += int(x['ok_boxed']); tot2[meta[pid]['level']] += 1
for n in (3, 9, 10):
    for pid, x in R.gen_rows(n).items():
        if not x['closed']:
            nc[meta[pid]['level']] += 1; nct[meta[pid]['level']] += 1
print('repair 1 alone by level:', {k: f'{lv2[k]} of {tot2[k]}' for k in sorted(lv2)}, 'sum', sum(lv2.values()), 'of', sum(tot2.values()),
      f'= {100 * sum(lv2.values()) / sum(tot2.values()):.1f}%')
print('never end with the final weights (attempts 3, 9, 10), by level:', dict(sorted(nc.items())), 'total', sum(nc.values()), 'of', sum(nct.values()))
# paired comparisons
def pair(a, b):
    A, B = R.gen_rows(a), R.gen_rows(b)
    return sum((not A[p]['ok_boxed']) and B[p]['ok_boxed'] for p in A), sum(A[p]['ok_boxed'] and not B[p]['ok_boxed'] for p in A)
# paired comparisons: (gained, lost) when going from the first attempt to the second
for a, b, g, l in ((2, 3, 13, 5), (8, 9, 28, 7), (11, 10, 20, 4), (12, 3, 20, 5), (1, 12, 27, 1), (2, 8, 4, 21)):
    check(f'from attempt {a} to {b}: gained, lost', pair(a, b), (g, l))
print('passes charged:', open(S + 'passes.txt').read().strip(), '| attempts:', open(S + 'attempts.txt').read().strip())
print('ALL NUMBERS MATCH' if ok else 'SOME NUMBERS DIFFER')
