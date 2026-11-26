"""Bound the upside of the final-phase runaways: in generations with no boxed answer, does the gold
answer string appear anywhere in the text (a crude sign the model at least considered it)?
Usage: python upside.py <gen.jsonl>"""
import sys, json, re
sys.path.insert(0, '/home/user/wsp14/work/scripts')
import lib
G = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
prac = {p['id']: p for p in lib.practice()}
BOX = chr(92) + 'boxed'
nobox = [g for g in G if BOX not in g['text']]
print(f'{len(nobox)} generations without any \\boxed (of {len(G)}): {sum(1 for g in nobox if g["finish"] == "stop")} ended, {sum(1 for g in nobox if g["finish"] == "length")} capped')
hit = []
for g in nobox:
    gold = prac[g['id']]['gold'].strip()
    t = g['text'].replace(' ', '').replace('\\!', '').replace('\\,', '').replace('\\left', '').replace('\\right', '')
    g2 = gold.replace(' ', '').replace('\\left', '').replace('\\right', '')
    # numeric gold: compare as plain number tokens
    found = g2 in t
    if not found:
        try:
            from fractions import Fraction
            m = re.fullmatch(r'-?\d+(\.\d+)?', gold)
            if m:
                val = float(Fraction(gold))
                cands = set(re.findall(r'(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])', t))
                found = any(abs(float(c) - val) < 1e-9 for c in cands)
        except Exception:
            pass
    if found:
        hit.append((g['id'], prac[g['id']]['level'][-1], gold, g['finish'], g['n_tokens']))
print(f'gold answer string appears somewhere in the text: {len(hit)} of {len(nobox)}')
for r in hit:
    print('   ', r)
# for comparison: in CORRECT generations the gold obviously appears; in wrong-but-boxed ones how often does gold appear in the text?
wrong = [g for g in G if BOX in g['text'] and not g['ok_boxed']]
def appears(g):
    gold = prac[g['id']]['gold'].strip(); t = g['text'].replace(' ', '')
    return gold.replace(' ', '') in t
print(f'for comparison: among {len(wrong)} wrong generations WITH a box, the gold string appears somewhere in {sum(1 for g in wrong if appears(g))}')
