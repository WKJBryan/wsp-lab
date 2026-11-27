"""Per-problem view of a pass, compared with the baseline pass: when did the thinking end, was the
answer right, what did the model write after </think>.
Usage: python a40_pass_summary.py <pass no> [base pass no=1] [examples]"""
import sys, json, re
import numpy as np
sys.path.insert(0, "/home/user/wsp14/work/analysis")
import tf

n = int(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 1
n_ex = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rows = {r["id"]: r for r in tf.gen_rows(n)}
base = {r["id"]: r for r in tf.gen_rows(nb)}
pr = {p["id"]: p for p in tf.practice()}
tok = tf.tokenizer()
s = json.load(open(f"{tf.STATE}/eval_pass_{n}.json"))
print({k: s[k] for k in ["accuracy_boxed", "accuracy_rescore", "closed", "cap_hit_rate", "mean_tokens", "by_level", "passes_spent", "remaining"]})
cut_after_close = sum(1 for r in rows.values() if r["closed"] and r["finish"] == "length")
print("closed but cut at the cap (long response):", cut_after_close)
toks = [r["n_tokens"] for r in rows.values()]
print("tokens: median", int(np.median(toks)), "mean", int(np.mean(toks)), "quartiles", np.quantile(toks, [0.25, 0.5, 0.75, 0.9]).round(0))
closed = [r for r in rows.values() if r["closed"]]
if closed:
    ct = []
    for r in closed:
        ids = tok.encode(r["text"])
        ct.append(ids.index(tf.T_THINK_END) if tf.T_THINK_END in ids else r["n_tokens"])
    print("closing token among closed generations: quartiles", np.quantile(ct, [0.1, 0.25, 0.5, 0.75, 0.9]).round(0), "n", len(ct))
    print("accuracy among closed:", round(100 * np.mean([r["ok_boxed"] for r in closed]), 1), "%  (boxed present:", sum("\\boxed" in r["text"] for r in closed), ")")
print("\ntransition table (baseline -> pass):")
tt = {}
for pid, r in rows.items():
    key = ("ok" if base[pid]["ok_boxed"] else ("closed" if base[pid]["closed"] else "cut"), "ok" if r["ok_boxed"] else ("closed" if r["closed"] else "cut"))
    tt[key] = tt.get(key, 0) + 1
for k in sorted(tt):
    print("  ", k, tt[k])
flips_up = [pid for pid, r in rows.items() if r["ok_boxed"] and not base[pid]["ok_boxed"]]
flips_dn = [pid for pid, r in rows.items() if not r["ok_boxed"] and base[pid]["ok_boxed"]]
print("newly correct:", len(flips_up), "lost:", len(flips_dn))
print("\nper problem:")
print(f"{'id':>4s} {'L':>1s} {'tok':>6s} {'cls':>3s} {'ok':>3s} {'rs':>3s} {'box':>3s} {'fin':>6s} {'think_tok':>9s} {'base':>5s} gold -> pred")
for pid, r in rows.items():
    ids = tok.encode(r["text"]) if r["closed"] else []
    tt_ = ids.index(tf.T_THINK_END) if ids and tf.T_THINK_END in ids else -1
    b = base[pid]
    print(f"{pid:>4s} {pr[pid]['level'][-1]} {r['n_tokens']:6d} {int(r['closed']):3d} {int(r['ok_boxed']):3d} {int(r['ok_rescored']):3d} {int('\\\\boxed' in r['text']):3d} {r['finish']:>6s} {tt_:9d} {('ok' if b['ok_boxed'] else ('cls' if b['closed'] else 'cut')):>5s} {pr[pid]['gold'][:22]!s:>22s} -> {str(r['pred_boxed'])[:22]!s}")
if n_ex:
    shown = 0
    for pid, r in rows.items():
        if r["closed"] and shown < n_ex:
            shown += 1
            t = r["text"]; i = t.index("</think>")
            print("=" * 110)
            print(f"{pid} {pr[pid]['level']} ok={r['ok_boxed']} gold={r['gold']!r} pred={r['pred_boxed']!r} think tokens={tok.encode(t[:i]).__len__()} finish={r['finish']}")
            print("PROBLEM:", pr[pid]["q"][:300])
            print("--- last 700 chars of the thinking ---"); print(t[max(0, i - 700): i])
            print("--- response (first 900 chars) ---"); print(t[i: i + 900])
