"""Text-level analysis of one pass's generations (CPU only, no GPU job).

usage: python analyze_gen.py <pass number> [--show ID[,ID...]] [--full ID]

Prints the summary, a failure taxonomy, and for every generation that was cut at the cap the
longest phrase that repeats (a "loop") and where it starts.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

STATE = Path.home() / "wsp14/sandbox/harness/state"
PRACTICE = json.loads((Path.home() / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))


def load(k):
    rows = [json.loads(l) for l in (STATE / f"gen_{k}.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["id"]: r for r in rows}


def longest_loop(text, min_len=12, max_len=4000):
    """Longest substring s of `text` that occurs at least 3 times back to back or scattered, with
    a greedy search: for a set of candidate lengths, find the substring that occurs most often
    and covers the most characters. Returns (phrase, count)."""
    # sentence/line split approach: find n-gram of lines that repeats
    lines = [l for l in text.split("\n")]
    best = ("", 0)
    # sliding windows of 1..12 lines
    for w in range(1, 13):
        seen = Counter()
        for i in range(0, len(lines) - w + 1):
            gram = "\n".join(lines[i:i + w])
            if len(gram.strip()) < 10:
                continue
            seen[gram] += 1
        for gram, c in seen.items():
            if c >= 3 and c * len(gram) > best[1] * len(best[0]):
                best = (gram, c)
    return best


def char_loop(text):
    """Exact character-level loop in the tail: the shortest period p (<= 600) such that the last
    L characters are periodic with period p, L >= 3p. Returns (p, L, unit) or None."""
    tail = text[-9000:]
    for p in range(2, 601):
        unit = tail[-p:]
        L = 0
        while L + p <= len(tail) and tail[-L - p:-L] == unit:
            L += p
        if L >= 3 * p:
            return p, L, unit
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("--show", default="")
    ap.add_argument("--full", default="")
    a = ap.parse_args()
    rows = load(a.k)
    meta = {p["id"]: p for p in PRACTICE}
    n = len(rows)
    cats = Counter()
    print(f"pass {a.k}: n={n}")
    for pid, r in rows.items():
        cut = r["finish"] == "length"
        closed = r["closed"]
        ok = r["ok_boxed"]
        has_box = r["pred_boxed"] is not None
        if cut:
            c = "cut@cap, closed" if closed else "cut@cap, thinking never closed"
        elif not closed:
            c = "stopped, never closed (no think end)"
        elif not has_box:
            c = "closed, no boxed answer"
        else:
            c = "closed, correct" if ok else "closed, wrong"
        cats[c] += 1
        r["_cat"] = c
        cl = char_loop(r["text"])
        r["_charloop"] = cl
    for c, v in cats.most_common():
        print(f"  {v:3d}  {c}")
    tok = sorted(r["n_tokens"] for r in rows.values())
    print("tokens: min", tok[0], "median", tok[n // 2], "max", tok[-1],
          "closed median", sorted(r["n_tokens"] for r in rows.values() if r["closed"])[max(0, sum(r['closed'] for r in rows.values()) // 2)])
    print()
    print("id | lvl | ok | finish | closed | ntok | char-loop(period,len) | category")
    for pid in rows:
        r = rows[pid]
        cl = r["_charloop"]
        cls = f"p={cl[0]},len={cl[1]}" if cl else "-"
        print(f"{pid:5s} | {meta[pid]['level'][-1]} | {int(r['ok_boxed'])} | {r['finish']:6s} | {int(r['closed'])} | {r['n_tokens']:6d} | {cls:16s} | {r['_cat']}")
    # loop table for cut ones
    print()
    print("=== generations cut at the cap ===")
    for pid in rows:
        r = rows[pid]
        if r["finish"] != "length":
            continue
        cl = r["_charloop"]
        lp, lc = longest_loop(r["text"])
        print(f"--- {pid} ({meta[pid]['level']}, {meta[pid]['type']}), ntok={r['n_tokens']}, closed={r['closed']}, "
              f"gold={meta[pid]['gold']}, boxed={r['pred_boxed']}, ok={r['ok_boxed']}")
        if cl:
            unit = cl[2]
            first = r["text"].rfind(unit * 3)
            print(f"    exact char loop: period {cl[0]} chars, tail length {cl[1]} chars; starts at char {first} of {len(r['text'])}")
            print("    unit:", repr(unit[:300]))
        else:
            print("    no exact character loop; repeated lines:", lc, "x", repr(lp[:200]))
    if a.show:
        for pid in a.show.split(","):
            r = rows[pid]
            t = r["text"]
            print(f"\n################ {pid} ({meta[pid]['level']}, {meta[pid]['type']}) gold={meta[pid]['gold']} "
                  f"pred={r['pred_boxed']} ok={r['ok_boxed']} finish={r['finish']} ntok={r['n_tokens']}")
            print("PROBLEM:", meta[pid]["q"])
            if a.full == pid or a.full == "all":
                print(t)
            else:
                print("--- head ---")
                print(t[:1500])
                print("--- tail ---")
                print(t[-2500:])


if __name__ == "__main__":
    main()
