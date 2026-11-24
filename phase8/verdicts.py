"""Phase 8 Parts B-D: accuracy and paired comparisons under both graders, with the
protocol's decision rules. Arms whose files do not exist yet are skipped.

Usage (the grader venv): python phase8/verdicts.py
"""

import gzip
import json
from pathlib import Path

import numpy as np

from analyze_saved import ARMS, load, paired, second_grade, validity_problems

HERE = Path(__file__).resolve().parent
COMPARE = [("b1", "E1"), ("b1", "O"), ("b3", "b1"), ("b2", "b1"), ("c1_final", "E1"),
           ("d_out", "E1"), ("d_in", "S"), ("c2_S", "S"), ("c2_O", "c2_E1"),
           # not pre-registered (added after seeing the arm accuracies); reported as exploratory
           ("c2_S", "E1"), ("b2", "E1"), ("b2", "O"), ("c1_final", "O"), ("c1_final", "b2")]


def arm(name):
    """(rescore.py verdicts, math-verify verdicts) for a carried run or a phase-8 run."""
    if name in ARMS:
        rows, texts = load(ARMS[name])
    else:
        run = json.loads((HERE / f"{name}.json").read_text())
        problems = validity_problems(run, name)
        assert not problems, f"{problems} (ISSUES #20: a run that failed or lacks a check is not a measurement)"
        rows, texts = run["rows"], json.loads(gzip.open(HERE / f"{name}_texts.json.gz", "rt", encoding="utf-8").read())
    return (np.array([bool(r["ok_rescored"]) for r in rows]),
            np.array([bool(second_grade(r["gold"], texts[r["id"]])) for r in rows]))


def verdict(p):
    d, (lo, hi) = p["delta"], p["ci95"]
    if abs(d) <= 4.0:
        return "MATCHES"
    if d < -4.0 and hi < 0:
        return "FALLS SHORT"
    if d > 4.0 and lo > 0:
        return "EXCEEDS"
    return "INCONCLUSIVE"


def main():
    arms, out = {}, dict(accuracy={}, pairs={})
    for a, b in COMPARE:
        if not all(n in ARMS or (HERE / f"{n}.json").exists() for n in (a, b)):
            continue
        for n in (a, b):
            if n not in arms:
                arms[n] = arm(n)
                out["accuracy"][n] = [round(100 * float(g.mean()), 1) for g in arms[n]]
        res = {}
        for label, i in (("rescore", 0), ("math_verify", 1)):
            p = paired(arms[a][i], arms[b][i], np.random.default_rng(0))
            res[label] = dict(p, verdict=verdict(p))
        out["pairs"][f"{a}_vs_{b}"] = res
        print(f"{a}_vs_{b}", json.dumps(res), flush=True)
    print("accuracy [rescore, math_verify]", json.dumps(out["accuracy"]))
    (HERE / "verdicts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
