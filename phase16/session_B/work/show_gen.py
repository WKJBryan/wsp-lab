"""Print generations of a pass to a file for reading.
usage: python show_gen.py <pass> <out file> [id[:maxchars][,id[:maxchars]...]] [--heads N] [--tails N]
"""
import json, sys
from pathlib import Path

STATE = Path.home() / "wsp14/sandbox/harness/state"
PRACTICE = {p["id"]: p for p in json.loads((Path.home() / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))}


def main():
    k, out = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else ""
    heads = 1500
    tails = 2500
    args = sys.argv[3:]
    if "--heads" in args:
        heads = int(args[args.index("--heads") + 1])
    if "--tails" in args:
        tails = int(args[args.index("--tails") + 1])
    rows = {}
    for i in (1,):
        pass
    for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        rows[r["id"]] = r
    with open(out, "w", encoding="utf-8") as f:
        ids = [w for w in which.split(",") if w] or list(rows)
        for w in ids:
            parts = w.split(":")
            pid = parts[0]
            mx = int(parts[1]) if len(parts) > 1 else None
            r = rows[pid]
            m = PRACTICE[pid]
            t = r["text"]
            f.write(f"\n################ {pid} ({m['level']}, {m['type']}) gold={m['gold']} pred={r['pred_boxed']} "
                    f"ok={r['ok_boxed']} finish={r['finish']} closed={r['closed']} ntok={r['n_tokens']} chars={len(t)}\n")
            f.write("PROBLEM: " + m["q"] + "\n")
            if mx is not None and len(t) > mx:
                f.write(f"--- head ({heads} chars) ---\n" + t[:heads] + "\n")
                mid = len(t) // 2
                f.write(f"--- middle (2000 chars around char {mid}) ---\n" + t[mid - 1000:mid + 1000] + "\n")
                f.write(f"--- tail ({tails} chars) ---\n" + t[-tails:] + "\n")
            else:
                f.write(t + "\n")


if __name__ == "__main__":
    main()
