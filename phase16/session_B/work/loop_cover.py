"""For the block loops of a pass: does the repeated block contain a line that starts with one of the flipped words? CPU.
usage: python loop_cover.py <pass>
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reads as R  # noqa: E402

WORDS = {"wait", "so", "maybe", "this", "let", "but", "is", "no", "if", "okay", "what", "yes", "then"}


def main():
    k = int(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(str(R.K.model_dir("base_model")))
    STRS = [s.replace("Ġ", " ").replace("Ċ", "\n") for s in tok.convert_ids_to_tokens(list(range(len(tok))))]
    rows = R.gen_rows(k)
    cov = dict(with_word=0, without_word=0, no_exact_loop=0)
    examples = []
    for pid, r in rows.items():
        if r["closed"]:
            continue
        g = tok(r["text"], add_special_tokens=False).input_ids
        per = R.token_period(g, max_period=4000, min_reps=6)
        if per is None:
            cov["no_exact_loop"] += 1
            continue
        unit = "".join(STRS[i] for i in g[per["start"]:per["start"] + per["period"]])
        firsts = []
        for line in unit.split("\n"):
            w = line.strip().lstrip("*-• ").strip()
            w = w.split(" ")[0] if w else ""
            firsts.append(w.strip("$,.?!'\"") if w else "")
        hit = any(f.lower() in WORDS for f in firsts)
        cov["with_word" if hit else "without_word"] += 1
        if not hit:
            examples.append((pid, unit[:110].replace("\n", "⏎")))
    print(f"pass {k}: {cov}")
    print("loops without a flipped line-start word:")
    for pid, u in examples:
        print(f"   {pid}: {u!r}")


if __name__ == "__main__":
    main()
