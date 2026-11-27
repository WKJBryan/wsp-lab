"""Trace decision rows of one sequence after lines that look like final statements.
usage: python trace_rows.py <prefix> <pass> <pid> [--regex RE] [--minp 0.005] [--all]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir()))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
IS_NL = np.array(["Ċ" in s for s in TOKS])
IS_WAIT = np.array([s.lstrip("Ġ*Ċ").startswith("Wait") for s in TOKS])
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]


def load(prefix, pid):
    d = np.load(f"{prefix}.{pid}.npz")
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("k", type=int)
    ap.add_argument("pid")
    ap.add_argument("--regex", default=r"answer is|final answer|ready|write (the|it|out)|so the answer|therefore|conclusion|final output|boxed")
    ap.add_argument("--minp", type=float, default=0.0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    d = load(a.prefix, a.pid)
    g = d["gids"].astype(np.int64)
    G = len(g)
    rows = R.gen_rows(a.k)
    cl = int(d["meta"][0])
    rx = re.compile(a.regex, re.I)
    nl = np.zeros(G, dtype=bool)
    nl[1:] = IS_NL[g[:-1]]
    wc = np.cumsum(IS_WAIT[g]) - IS_WAIT[g]
    text = "".join(STRS[i] for i in g)
    # char offsets of tokens
    off = np.cumsum([0] + [len(STRS[i]) for i in g])
    n_shown = 0
    for t in np.where(nl)[0]:
        if cl and t >= len(g) - 1:
            break
        line = text[off[max(0, t - 60)]:off[t]].split("\n")[-2] if text[off[t - 1]] == "\n" else ""
        line = [l for l in text[off[0]:off[t]].split("\n") if l.strip()][-1] if t > 0 else ""
        if not a.all and not rx.search(line):
            continue
        if d["prof_think_p"][t] < a.minp:
            continue
        top = d["prof_top"][t]
        topl = d["prof_topl"][t]
        pk = d["prof_pk"][t]
        pkl = d["prof_pkl"][t]
        print(f"t={t:5d} W={wc[t]:4d} </think> {d['prof_think'][t]:5.2f} (p={d['prof_think_p'][t]:.3f}, rank {d['prof_think_rank'][t]:2d}) "
              f"next={STRS[g[t]]!r:10s} raw: " + " ".join(f"{STRS[i]!r}:{v:.1f}" for i, v in zip(top[:4], topl[:4])) +
              " | pen: " + " ".join(f"{STRS[i]!r}:{v:.1f}" for i, v in zip(pk[:3], pkl[:3])) + f"\n        after line: {line.strip()[-90:]!r}")
        n_shown += 1
        if n_shown >= 60:
            break


if __name__ == "__main__":
    main()
