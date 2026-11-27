"""Print the tail of the thinking (before </think>) and the answer head for selected problems of a pass.
usage: python tail_gen.py <pass> <out file> <id[,id...]|closed|cut> [--n 12] [--chars 1800]
"""
import json
import sys
from pathlib import Path

STATE = Path.home() / "wsp14/sandbox/harness/state"
PRACTICE = {p["id"]: p for p in json.loads((Path.home() / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))}


def main():
    k, out, which = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 12
    chars = int(sys.argv[sys.argv.index("--chars") + 1]) if "--chars" in sys.argv else 1800
    rows = {}
    for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        rows[r["id"]] = r
    if which == "closed":
        ids = [p for p, r in rows.items() if r["closed"]]
    elif which == "cut":
        ids = [p for p, r in rows.items() if not r["closed"]]
    elif which == "closedwrong":
        ids = [p for p, r in rows.items() if r["closed"] and not r["ok_boxed"]]
    else:
        ids = which.split(",")
    with open(out, "w", encoding="utf-8") as f:
        for pid in ids[:n]:
            r = rows[pid]
            m = PRACTICE[pid]
            t = r["text"]
            f.write(f"\n################ {pid} ({m['level']}, {m['type']}) gold={m['gold']} pred={r['pred_boxed']} ok={r['ok_boxed']} "
                    f"finish={r['finish']} ntok={r['n_tokens']} closed={r['closed']}\nPROBLEM: {m['q'][:600]}\n")
            if r["closed"]:
                i = t.index("</think>")
                f.write(f"--- thinking tail ({chars} chars) ---\n{t[max(0, i - chars):i]}</think>\n--- answer ---\n{t[i + 8:i + 8 + 1500]}\n")
            else:
                f.write(f"--- tail ({chars} chars) ---\n{t[-chars:]}\n")


if __name__ == "__main__":
    main()
