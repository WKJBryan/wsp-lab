"""Find the think-end tag offsets in closed generations and print the context before closure.
Also print the top next-tokens token-ids there for later lens runs."""
import json, sys

TAG = '<' + '|end' + 'of' + 'thinking|' + '>'   # assembled to avoid literal marker in source listings
rows = {r['id']: r for r in (json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_1.jsonl', encoding='utf-8'))}
for gid in (sys.argv[1:] or ['p28', 'p66', 'p101', 'p103', 'p105', 'p124']):
    t = rows[gid]['text']
    idx = t.find(TAG)
    print(f'== {gid}: closed={rows[gid]["closed"]} ok={rows[gid]["ok_boxed"]} tag_offset={idx} len={len(t)}')
    if idx >= 0:
        print('   before tag:', repr(t[max(0, idx - 300):idx]))
        print('   after tag:', repr(t[idx:idx + 200]))
