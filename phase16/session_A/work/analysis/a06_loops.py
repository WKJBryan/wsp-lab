"""Loop anatomy of a pass: for every cut-off generation find the period of the final verbatim
loop, where it starts, and what the model wrote right before it.
Usage: python a06_loops.py <pass no> [n_examples]
CPU only."""
import sys, json, collections
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1])
n_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = tf.gen_rows(n)
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()

def final_period(ids, K=60, tol=0):
    """Smallest P such that the last K tokens repeat with period P (allow `tol` mismatches)."""
    for P in range(1, 2000):
        if len(ids) < K + P:
            break
        mism = sum(1 for i in range(1, K + 1) if ids[-i] != ids[-i - P])
        if mism <= tol:
            return P
    return None

def loop_start(ids, P, max_lead=20000):
    """Earliest index s such that ids[s:] is P-periodic until the end (with <=1% mismatches per 200)."""
    s = len(ids) - 1
    # walk backwards while the period holds
    j = len(ids) - 1
    run = 0
    best = len(ids)
    mism_run = 0
    i = len(ids) - 1 - P
    last_ok = len(ids) - 1
    while i - P >= 0:
        if ids[i] == ids[i - P]:
            last_ok = i - P
            run += 1
            mism_run = 0
        else:
            mism_run += 1
            if mism_run > 6:      # more than a few consecutive mismatches: the loop is over
                break
        i -= 1
    return last_ok

info = []
for r in rows:
    ids = tok.encode(r["text"])
    P = final_period(ids)
    d = dict(id=r["id"], n=len(ids), finish=r["finish"], closed=r["closed"], ok=r["ok_boxed"], period=P)
    if P and r["finish"] == "length":
        s = loop_start(ids, P)
        d["loop_start"] = s
        d["loop_frac"] = (len(ids) - s) / len(ids)
        d["reps"] = (len(ids) - s) / P
    info.append(d)
    print(f"{r['id']:>4s} n={len(ids):6d} {r['finish']:6s} closed={int(r['closed'])} ok={int(r['ok_boxed'])} period={P} "
          + (f"loop_start={d['loop_start']:6d} ({100 * d['loop_frac']:.0f}% of text) reps={d['reps']:.0f}" if P and r["finish"] == "length" else ""))
json.dump(info, open(f"/home/user/wsp14/work/cache/loops_pass{n}.json", "w"))

cut = [d for d in info if d["finish"] == "length"]
per = [d["period"] for d in cut if d["period"]]
print("\ncut-off:", len(cut), "| with a verbatim final loop:", len(per), "| period min/median/max:", min(per), sorted(per)[len(per) // 2], max(per))
starts = sorted(d["loop_start"] for d in cut if d["period"])
print("loop start tokens: min", starts[0], "quartiles", starts[len(starts) // 4], starts[len(starts) // 2], starts[3 * len(starts) // 4], "max", starts[-1])
nop = [d["id"] for d in cut if not d["period"]]
print("cut-off without a clean period:", nop)

if n_ex:
    shown = 0
    for r, d in zip(rows, info):
        if d["period"] and r["finish"] == "length" and shown < n_ex:
            shown += 1
            ids = tok.encode(r["text"]); s = d["loop_start"]
            print("=" * 110)
            print(f"{r['id']}  {pr[r['id']]['level']}  period={d['period']}  loop_start={s}  gold={r['gold']}")
            print("PROBLEM:", pr[r["id"]]["q"][:400])
            print("--- 700 tokens before the loop starts ---")
            print(tok.decode(ids[max(0, s - 700): s + 2 * d["period"]]))
