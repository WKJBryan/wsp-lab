"""Print what follows the first </think> in traces of a pass (to study the answer phase). CPU.
usage: python after_think.py <pass> <out file> <id[,id...]|closedcut> [--head 2500] [--tail 1500]
"""
import json
import sys
from pathlib import Path

STATE = Path.home() / "wsp14/sandbox/harness/state"
PRACTICE = {p["id"]: p for p in json.loads((Path.home() / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))}


def main():
    k, out, which = sys.argv[1], sys.argv[2], sys.argv[3]
    head = int(sys.argv[sys.argv.index("--head") + 1]) if "--head" in sys.argv else 2500
    tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 1500
    rows = {}
    for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        rows[r["id"]] = r
    ids = which.split(",")
    if which == "closedcut":
        ids = [p for p, r in rows.items() if r["closed"] and r["finish"] == "length"]
    elif which == "closedok":
        ids = [p for p, r in rows.items() if r["closed"] and r["finish"] == "stop"]
    with open(out, "w", encoding="utf-8") as f:
        for pid in ids:
            r = rows[pid]
            m = PRACTICE[pid]
            t = r["text"]
            f.write(f"\n################ {pid} ({m['level']}, {m['type']}) gold={m['gold']} pred_boxed={r['pred_boxed']} ok={r['ok_boxed']} "
                    f"finish={r['finish']} ntok={r['n_tokens']} chars={len(t)} boxed_count={t.count(chr(92) + 'boxed')}\n")
            if "</think>" in t:
                i = t.index("</think>")
                before = t[:i]
                j = t.find("</think>", i + 8)
                f.write(f"--- last 500 chars of thinking ---\n{before[-500:]}\n--- answer phase: first {head} chars ---\n{t[i + 8:i + 8 + head]}\n")
                if len(t) - i > head + 8:
                    f.write(f"--- tail ({tail} chars) ---\n{t[-tail:]}\n")
                f.write(f"--- number of </think> in text: {t.count('</think>')}; <|im_end|> present: {'<|im_end|>' in t}\n")


if __name__ == "__main__":
    main()
