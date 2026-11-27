"""Potential of a response-phase fix: in the responses that never finished, how often does the model
write an explicit answer statement (and does it contain the gold answer)? Counted per pass.
Usage: python a51_rescue_potential.py <pass no>"""
import sys, re
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()


def norm(s):
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s).replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "").strip("$")
    return s


ANS = re.compile(r"(answer|therefore|thus|hence|so the|so, the|in conclusion|conclusion|final result|the value is|the value of .* is|is \\boxed)", re.I)
stuck = [r for r in rows if r["closed"] and r["finish"] == "length"]
tot = dict(stuck=len(stuck), ans_stmt=0, ans_gold=0, ans_gold_first=0, gold_any=0, loop_like=0)
ex = []
for r in stuck:
    t = r["text"]; resp = t[t.index("</think>"):]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s*\n+", resp) if len(s.strip()) > 8]
    gold = norm(pr[r["id"]]["gold"])
    ans = [s for s in sents if ANS.search(s) and not re.search(r"(wait|maybe|if |could|would|question|ask)", s, re.I)]
    g = [s for s in ans if gold and gold in norm(s)]
    tot["ans_stmt"] += bool(ans); tot["ans_gold"] += bool(g)
    tot["gold_any"] += bool(gold) and (gold in norm(resp[:12000]))
    # loop-like: some 8-gram repeated many times
    words = resp.split()
    from collections import Counter
    c = Counter(" ".join(words[i:i + 8]) for i in range(0, max(0, len(words) - 8), 4))
    tot["loop_like"] += c.most_common(1)[0][1] >= 30 if c else False
    if g and len(ex) < 14:
        ex.append((r["id"], pr[r["id"]]["level"], pr[r["id"]]["gold"], g[0][:140]))
print("pass", n, tot)
print("examples of explicit answer statements (without doubt words) in stuck responses that contain the gold answer:")
for e in ex:
    print("  ", e)
