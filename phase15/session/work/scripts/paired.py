"""Paired per-problem comparison of two passes.
Usage: python paired.py <gen_a.jsonl> <gen_b.jsonl>"""
import sys, json
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
A = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[1], encoding='utf-8')}
B = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[2], encoding='utf-8')}
prac = {p['id']: p for p in lib.practice()}
gain = [k for k in A if not A[k]['ok_boxed'] and B[k]['ok_boxed']]
loss = [k for k in A if A[k]['ok_boxed'] and not B[k]['ok_boxed']]
both_bad = [k for k in A if not A[k]['ok_boxed'] and not B[k]['ok_boxed']]
print(f'A correct {sum(A[k]["ok_boxed"] for k in A)}, B correct {sum(B[k]["ok_boxed"] for k in B)}; gained {len(gain)}, lost {len(loss)}, both wrong {len(both_bad)}, both right {len(A) - len(gain) - len(loss) - len(both_bad)}')
st = lambda g: 'stop' if g['finish'] == 'stop' else 'cap'
print('\ngained (wrong in A -> right in B): id level | A: end, tokens, pred | B: end, tokens, pred | gold')
for k in sorted(gain, key=lambda s: int(s[1:])):
    print(f'  {k:5s} L{prac[k]["level"][-1]} | A {st(A[k]):4s} {A[k]["n_tokens"]:6d} {str(A[k]["pred_boxed"])[:22]:22s} | B {st(B[k]):4s} {B[k]["n_tokens"]:6d} {str(B[k]["pred_boxed"])[:22]:22s} | {prac[k]["gold"][:22]}')
print('\nlost (right in A -> wrong in B):')
for k in sorted(loss, key=lambda s: int(s[1:])):
    print(f'  {k:5s} L{prac[k]["level"][-1]} | A {st(A[k]):4s} {A[k]["n_tokens"]:6d} {str(A[k]["pred_boxed"])[:22]:22s} | B {st(B[k]):4s} {B[k]["n_tokens"]:6d} {str(B[k]["pred_boxed"])[:22]:22s} | {prac[k]["gold"][:22]}')
# outcome-type transitions
import collections
def typ(g):
    if g['finish'] == 'stop':
        return 'stopped'
    return 'closed-capped' if '</think>' in g['text'] else 'never-closed'
tr = collections.Counter((typ(A[k]), typ(B[k])) for k in A)
print('\noutcome-type transitions A -> B:')
for (a, b), c in sorted(tr.items(), key=lambda kv: -kv[1]):
    print(f'  {a:14s} -> {b:14s}: {c}')
