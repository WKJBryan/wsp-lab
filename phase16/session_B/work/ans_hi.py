"""Where in the answer phase is the model's own </think> logit high? CPU.
usage: python ans_hi.py
"""
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as PR  # noqa: E402
import reads as R  # noqa: E402

WORK = Path.home() / "wsp14/work"
STRS = PR.STRS


def main():
    n_ans = 0; n_hi = 0; at_end = 0; rows = []
    for pid in R.gen_rows(3):
        d = np.load(WORK / f"r3/xR1.{pid}.npz")
        pos, think, top, region, plen = d["pos"], d["think"], d["top"], d["region"], int(d["plen"])
        if d["meta"][0] != 1:
            continue
        ans = np.where(region == 2)[0]
        if len(ans) == 0:
            continue
        last = pos[ans[-1]]
        ids = None
        for i in ans:
            n_ans += 1
            if think[i] >= 20.5:
                n_hi += 1
                if pos[i] >= last - 12:
                    at_end += 1
                rows.append((pid, int(pos[i]), int(last), float(think[i]), float(top[i])))
    print(f"answer-phase sampled rows {n_ans}; with own </think> logit >= 20.5: {n_hi} ({100 * n_hi / n_ans:.3f}%), of which within 12 positions of the end: {at_end}")
    print("pid, position, last answer position, </think> logit, top-1 logit")
    for r in rows[:40]:
        print("  ", r)
    # context of a few
    tok_ids = {}
    for r in rows[:6]:
        pid = r[0]
        d = np.load(WORK / f"r3/xR1.{pid}.npz")
        rows_ = R.gen_rows(3)
        g = tok_encode(rows_[pid]["text"])
        p = r[1] - int(d["plen"])
        print(f"\n{pid} at generated token {p} of {len(g)}: ...{''.join(STRS[i] for i in g[max(0, p - 90):p + 1])[-220:]!r}")


def tok_encode(text):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
    return tok(text, add_special_tokens=False).input_ids


if __name__ == "__main__":
    main()
