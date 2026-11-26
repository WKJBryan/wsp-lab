"""Show excerpts of generations: python show.py <gen.jsonl> <id> [nchars_head] [n_mid] [nchars_tail]"""
import json, sys
rows = {json.loads(l)['id']: json.loads(l) for l in open(sys.argv[1], encoding='utf-8')}
prac = {p['id']: p for p in json.load(open('/home/user/wsp14/sandbox/practice.json', encoding='utf-8'))}
r = rows[sys.argv[2]]
head = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
nmid = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tail = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
t = r['text']
print('PROBLEM:', prac[r['id']]['q'][:1500])
print('GOLD:', prac[r['id']]['gold'], '| ntok', r['n_tokens'], '| finish', r['finish'], '| chars', len(t))
print('\n===== HEAD =====\n', t[:head])
L = len(t)
for i in range(nmid):
    a = head + (L - head - tail) * (i + 1) // (nmid + 1)
    print('\n===== MID at %d (%.0f%%) =====\n' % (a, 100*a/L), t[a:a+1200])
print('\n===== TAIL =====\n', t[-tail:])
