"""Phase-14 practice pool: 128 problems from the MATH training split.

Seeded and fixed before the session. Disjoint from the scored set (MATH-500 is drawn from the
test split; any training problem whose text equals a scored problem is dropped). The level mix
mirrors MATH-500's. The editor sees the questions and the gold answers; never the solutions.

Usage: uv run python phase14/make_practice.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
SEED, N = 20261124, 128
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")   # the dataset stores one config per subject


def main():
    train = [row for subject in SUBJECTS
             for row in load_dataset("HuggingFaceH4/MATH", subject, split="train")]
    scored = {re.sub(r"\s+", "", p["q"]) for p in json.loads((HERE.parent / "phase3c" / "problems.json").read_text(encoding="utf-8"))}
    by_level = {}
    dropped = 0
    for row in train:
        if re.sub(r"\s+", "", row["problem"]) in scored:
            dropped += 1
            continue
        gold = (BOXED.findall(row["solution"]) or [None])[-1]
        if gold is None:
            continue
        by_level.setdefault(row["level"], []).append(dict(q=row["problem"], gold=gold, type=row["type"]))
    # MATH-500's level mix, scaled to 128 (L1 43, L2 90, L3 105, L4 128, L5 134 of 500)
    share = Counter({"Level 1": 43, "Level 2": 90, "Level 3": 105, "Level 4": 128, "Level 5": 134})
    quota = {k: round(v / 500 * N) for k, v in share.items()}
    rng = random.Random(SEED)
    pool = []
    for level in sorted(quota):
        rows = sorted(by_level[level], key=lambda r: r["q"])
        pool += [dict(r, level=level) for r in rng.sample(rows, quota[level])]
    rng.shuffle(pool)
    out = [dict(id=f"p{i}", **r) for i, r in enumerate(pool)]
    (HERE / "practice.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(train_rows=len(train), dropped_as_scored=dropped, pool=len(out),
                          by_level=dict(sorted(Counter(r["level"] for r in out).items())),
                          by_type=dict(sorted(Counter(r["type"] for r in out).items()))), ensure_ascii=False))


if __name__ == "__main__":
    main()
