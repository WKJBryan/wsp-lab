"""Compact per-group tables from out/wstats.json."""
import json, re, sys
rows = json.load(open('/home/user/wsp14/work/out/wstats.json'))
groups = {}
for k, d in rows.items():
    groups.setdefault(re.sub(r'layers\.\d+\.', 'layers.L.', k), []).append(k)
want = sys.argv[1:] 
for suf, ks in groups.items():
    if want and not any(w in suf for w in want): continue
    print('==', suf.replace('model.language_model.', ''))
    for k in sorted(ks, key=lambda s: int(re.search(r'layers\.(\d+)\.', s).group(1)) if 'layers.' in s else -1):
        d = rows[k]
        name = re.sub(r'model\.language_model\.layers\.', '', k)
        f = lambda v: '-' if v is None else '/'.join('%g' % x for x in v)
        print('%-34s std %.4f kurt %6.1f min %.3f max %.3f row %s col %s sv %s' % (
            name, d['std'], d['kurt'], d['mn'], d['mx'], f(d.get('rownorm')), f(d.get('colnorm')), f(d.get('s'))))
