"""Show the thinking just before a simulated close row (from val_E2 output) for selected problems. CPU.
usage: python sim_close_view.py <pass> <out file> <out of val_E2> [--chars 900] [--only-closed]
"""
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
TOKS = tok.convert_ids_to_tokens(list(range(len(tok))))
STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in TOKS]
meta = {p["id"]: p for p in R.K.PRACTICE}


def main():
    k, out, val = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    chars = int(sys.argv[sys.argv.index("--chars") + 1]) if "--chars" in sys.argv else 900
    only_closed = "--only-closed" in sys.argv
    rows = R.gen_rows(k)
    sim = {}
    for line in Path(val).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(p\d+)\s+\| (\d) (\d) L(\d) \| +([\d-]+) \| +(\d+|None) ([\d.]+) ", line)
        if m and m.group(6) != "None":
            sim[m.group(1)] = (int(m.group(6)), float(m.group(7)), int(m.group(2)), int(m.group(3)))
    with open(out, "w", encoding="utf-8") as f:
        for pid, (row, p, cl, ok) in sim.items():
            if only_closed and not cl:
                continue
            if not only_closed and cl:
                continue
            r = rows[pid]
            g = tok(r["text"], add_special_tokens=False).input_ids
            before = "".join(STRS[i] for i in g[:row])
            ans = ""
            if cl:
                i = r["text"].index("</think>")
                ans = r["text"][i + 8:i + 8 + 700]
            f.write(f"\n################ {pid} ({meta[pid]['level']}, {meta[pid]['type']}) gold={meta[pid]['gold']} | pass result ok={ok} "
                    f"closed at {r['n_tokens']} pred={r['pred_boxed']} | simulated close at {row} with P={p:.2f}\n"
                    f"PROBLEM: {meta[pid]['q'][:400]}\n--- thinking before the simulated close ---\n{before[-chars:]}\n")
            if cl:
                f.write(f"--- answer written in the pass ---\n{ans}\n")


if __name__ == "__main__":
    main()
