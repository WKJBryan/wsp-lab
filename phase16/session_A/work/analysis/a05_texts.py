"""Text-level look at one pass: what do the generations look like, where do they fail?
Usage: python a05_texts.py <pass no> [max chars per sample]
CPU only; reads harness/state/gen_<pass>.jsonl."""
import sys, json, re, collections
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
show = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
HEDGES = ["wait", "hmm", "but let", "double-check", "double check", "alternatively", "instead", "let me check", "let me verify", "another way", "actually", "hold on", "let me reconsider", "reconsider", "let me re-examine", "re-examine", "let me recheck", "recheck"]

def sent_set(text):
    sents = [s.strip().lower() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 25]
    c = collections.Counter(sents)
    return c

print(f"{'id':>4s} {'L':>1s} {'tok':>6s} {'fin':>7s} {'cls':>3s} {'ok':>3s} {'rs':>3s} {'box':>3s} {'cjk%':>5s} {'wait/k':>6s} {'repSent':>7s} type")
tot = collections.Counter()
for r in rows:
    t = r["text"]; k = len(t) / 1000
    cjk = sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af")
    low = t.lower()
    waits = sum(low.count(h) for h in HEDGES)
    c = sent_set(t)
    rep = c.most_common(1)[0][1] if c else 0
    lvl = pr[r["id"]]["level"][-1]
    print(f"{r['id']:>4s} {lvl:>1s} {r['n_tokens']:6d} {r['finish']:>7s} {int(r['closed']):3d} {int(r['ok_boxed']):3d} {int(r['ok_rescored']):3d} {int('\\\\boxed' in t):3d} {100 * cjk / max(1, len(t)):5.1f} {waits / max(k, 0.001):6.1f} {rep:7d} {pr[r['id']]['type'][:18]}")
    tot["n"] += 1; tot["ok"] += r["ok_boxed"]; tot["closed"] += r["closed"]; tot["cap"] += r["finish"] == "length"
    tot["cjk"] += cjk / max(1, len(t)) > 0.02
print(dict(tot))
if show:
    for r in rows[:show]:
        print("=" * 100)
        print(r["id"], pr[r["id"]]["level"], r["n_tokens"], r["finish"], r["closed"], r["ok_boxed"], "gold", r["gold"], "pred", r["pred_boxed"])
        print(pr[r["id"]]["q"])
        print("-" * 40)
        print(r["text"][:6000])
        print("...")
        print(r["text"][-4000:])
