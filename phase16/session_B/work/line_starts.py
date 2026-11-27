"""Line-start tokens inside the remaining loop regions (and in ordinary text) of a pass. CPU.
usage: python line_starts.py <pass> [--top 40]
"""
import collections
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]


def main():
    k = int(sys.argv[1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    rows = R.gen_rows(k)
    loop_c = collections.Counter(); early_c = collections.Counter(); n_loop = 0; n_early = 0
    for pid, r in rows.items():
        if r["closed"]:
            continue
        g = tok(r["text"], add_special_tokens=False).input_ids
        per = R.token_period(g, max_period=4000, min_reps=6)
        if per is None:
            continue
        # loop region: from the first repetition of the periodic tail to the end
        lo = per["start"] + per["period"] * 2
        for name, ids, c in (("loop", g[lo:], loop_c), ("early", g[1500:4000], early_c)):
            # first non-space token of each line
            at_start = True
            for i in ids:
                s = STRS[i]
                if "\n" in s:
                    at_start = True
                    s = s.split("\n")[-1]
                if at_start and s.strip():
                    c[s.strip() if len(s.strip()) <= 12 else s.strip()[:12]] += 1
                    at_start = False
    tl, te = sum(loop_c.values()), sum(early_c.values())
    print(f"pass {k}: line starts in loop regions {tl}, in early text {te}")
    print(f"{'token':16s} {'loop%':>7s} {'early%':>7s}  ratio")
    cum = 0
    for t, c in loop_c.most_common(top):
        cum += c
        print(f"{t!r:16s} {100 * c / tl:7.2f} {100 * early_c[t] / te:7.2f}  {c / tl / max(early_c[t] / te, 1e-9):6.1f}   cum {100 * cum / tl:5.1f}%")


if __name__ == "__main__":
    main()
