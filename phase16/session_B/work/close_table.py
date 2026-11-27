"""Every close of a pass: the last thinking line before </think>, whether the answer phase finished, correctness.
usage: python close_table.py <pass> <out file>
"""
import json
import sys
from pathlib import Path

import numpy as np

STATE = Path.home() / "wsp14/sandbox/harness/state"
PRACTICE = {p["id"]: p for p in json.loads((Path.home() / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))}


def main():
    k, out = sys.argv[1], sys.argv[2]
    rows = {}
    for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        rows[r["id"]] = r
    groups = {}
    with open(out, "w", encoding="utf-8") as f:
        for pid, r in rows.items():
            if not r["closed"]:
                continue
            t = r["text"]
            i = t.index("</think>")
            th = t[:i]
            ls = [l.strip() for l in th.split("\n") if l.strip()]
            last = ls[-1] if ls else ""
            after = t[i + 8:]
            clean = r["finish"] == "stop"
            low = last.lower()
            kind = ("answer-is" if ("answer is" in low or "answer:" in low or "final answer" in low or "answer should be" in low or "answer seems" in low
                                    or "answer would be" in low or "answer:" in low) else
                    "result-line" if any(w in low for w in (" = ", " is ", "result", "sum", "total", "count")) and not low.endswith("?") else
                    "affirm" if any(w in low for w in ("correct", "solid", "good", "holds", "robust", "consistent", "yes.", "satisf", "seems", "valid", "done", "ready",
                                                       "proceed", "stop", "confident", "stick", "write", "output", "response", "check")) else "other")
            key = (kind, "clean" if clean else "leaky")
            groups.setdefault(key, []).append(pid)
            f.write(f"{pid:5s} L{PRACTICE[pid]['level'][-1]} ok={int(r['ok_boxed'])} {'CLEAN' if clean else 'LEAKY'} ntok={r['n_tokens']:6d} "
                    f"think_tok~{int(len(th) / 3.6):6d} boxed={r['pred_boxed']!r:26.26s} | {last[-90:]!r}\n")
        f.write("\nsummary (kind, answer phase): n problems, n correct\n")
        for key in sorted(groups):
            ps = groups[key]
            f.write(f"   {key}: {len(ps)} problems, {sum(int(rows[p]['ok_boxed']) for p in ps)} correct  {ps}\n")


if __name__ == "__main__":
    main()
