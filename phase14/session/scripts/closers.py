import json

rows = [json.loads(l) for l in open('/home/user/wsp14/sandbox/harness/state/gen_2.jsonl')]
closed = [r for r in rows if r['closed']]
print('closed gens:', len(closed), ' of which ok:', sum(r['ok_boxed'] for r in closed))
for r in closed:
    print('  %s ntok=%5d fin=%s ok=%d pred=%r gold=%r' % (
        r['id'], r['n_tokens'], r['finish'][:4], int(r['ok_boxed']),
        str(r['pred_boxed'])[:24], str(r['gold'])[:16]))
notc = [r for r in rows if not r['closed'] and r['finish'] != 'length']
print('not closed but finished:', [(r['id'], r['n_tokens']) for r in notc])
