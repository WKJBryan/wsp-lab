"""For a pass: outcome structure. How many closed, how many responses finished, how the response's
chance to finish depends on when the thinking was closed and on the difficulty level.
Usage: python a50_outcomes.py <pass no> [reference pass no]"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); ref = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
base = {r["id"]: r for r in tf.gen_rows(ref)}
tok = tf.tokenizer()
cl = []
for r in rows:
    if r["closed"]:
        g = tok.encode(r["text"]); te = g.index(tf.T_THINK_END)
        cl.append((r["id"], te, r["finish"] == "stop", r["ok_boxed"], int(pr[r["id"]]["level"][-1]), r["n_tokens"] - te, "  \\boxed" in r["text"]))
te = np.array([c[1] for c in cl]); fin = np.array([c[2] for c in cl]); ok = np.array([c[3] for c in cl]); lv = np.array([c[4] for c in cl])
print(f"pass {n}: correct {sum(r['ok_boxed'] for r in rows)}/128 | closed {len(cl)} | response finished {int(fin.sum())} (correct {int(ok.sum())}) | "
      f"closed but response cut {int((~fin).sum())} | never closed {128 - len(cl)}")
if len(cl):
    print("thinking length at closing (tokens): deciles", np.quantile(te, [0.1, 0.25, 0.5, 0.75, 0.9]).round(0))
    print("response length when finished (tokens): deciles", np.quantile([c[5] for c in cl if c[2]], [0.1, 0.5, 0.9]).round(0) if fin.any() else None)
    for name, sel in [("< 400", te < 400), ("400-800", (te >= 400) & (te < 800)), ("800-2000", (te >= 800) & (te < 2000)),
                      ("2000-4000", (te >= 2000) & (te < 4000)), ("4000-7000", (te >= 4000) & (te < 7000)), (">= 7000", te >= 7000)]:
        if sel.sum():
            print(f"  closed at {name:9s}: n={int(sel.sum()):2d}  response finished {int(fin[sel].sum()):2d}  correct {int(ok[sel].sum()):2d}")
    for L in "12345":
        s = lv == int(L)
        nc = sum(1 for r in rows if not r["closed"] and pr[r["id"]]["level"][-1] == L)
        tot = sum(1 for p in pr.values() if p["level"][-1] == L)
        print(f"  Level {L}: problems {tot:2d} | closed {int(s.sum()):2d} | finished {int(fin[s].sum()):2d} | correct {int(ok[s].sum()):2d} | never closed {nc:2d}")
tt = {}
for r in rows:
    b = base[r["id"]]
    key = ("ok" if b["ok_boxed"] else "not"), ("ok" if r["ok_boxed"] else "not")
    tt[key] = tt.get(key, 0) + 1
print(f"versus pass {ref}:", tt, "| newly correct", tt.get(("not", "ok"), 0), "lost", tt.get(("ok", "not"), 0))
