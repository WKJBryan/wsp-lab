"""Phase 8 Part A: re-analysis of saved generations (no GPU). Rules: phase8/protocol.md.

Per arm: accuracy overall / among finished / among cut-off, mean tokens of finished.
Pairs: exact McNemar + paired-bootstrap 95% CI (10,000 resamples, numpy seed 0).
Second grader: math-verify on gold + full output; exceptions are counted, never
replaced by string comparison.

Usage (a Linux python with numpy, scipy, math-verify): python phase8/analyze_saved.py
"""

import gzip
import json
from pathlib import Path

import numpy as np
from math_verify import parse, verify
from math_verify.errors import TimeoutException  # not an Exception subclass
from scipy.stats import binomtest

P3C = Path(__file__).resolve().parent.parent / "phase3c"
ARMS = dict(S="p3c_stock", O="p3c_c3_original", E1="p4_mimo_pro", E2="p4_glmflash", R="p4_c2_random")
PAIRS = [("E1", "S"), ("E1", "O"), ("E2", "E1"), ("O", "S")]


def validity_problems(data, name):
    """Why a run is not a measurement: its recorded problems, or a check that is not on record."""
    v = data["summary"].get("validity") or {}
    problems = list(v.get("problems") or [])
    if not v.get("soup_judged"):
        problems.append("no fragment-tail check on record")
    if not v.get("admission_tested"):
        problems.append("no batch-position check on record")
    return [f"{name}: {p}" for p in problems]


def load(name):
    data = json.loads((P3C / f"{name}_rescored.json").read_text())
    problems = validity_problems(data, name)
    assert not problems, f"{problems} (ISSUES #20: a run that failed or lacks a check is not a measurement)"
    rows = data["rows"]
    plain = P3C / f"{name}_texts.json"
    texts = json.loads(plain.read_text() if plain.exists()
                       else gzip.open(f"{plain}.gz", "rt", encoding="utf-8").read())
    return rows, texts


def pct(x):
    return round(100 * float(np.mean(x)), 1) if len(x) else None


def second_grade(gold, text):
    """True/False, or None when math-verify cannot grade the row."""
    try:
        g = parse(f"${gold}$", raise_on_error=True)
        if not g:
            return None
        return bool(verify(g, parse(text, raise_on_error=True), raise_on_error=True))
    except (Exception, TimeoutException):
        return None


def paired(a, b, rng):
    n = len(a)
    idx = rng.integers(0, n, size=(10_000, n))
    lo, hi = np.percentile((a[idx].mean(1) - b[idx].mean(1)) * 100, [2.5, 97.5])
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return dict(delta=round((a.mean() - b.mean()) * 100, 1), ci95=[round(lo, 1), round(hi, 1)],
                only_first=only_a, only_second=only_b, mcnemar_p=float(f"{p:.3g}"))


def main():
    out, ok, ok2 = {}, {}, {}
    for arm, name in ARMS.items():
        rows, texts = load(name)
        ids = [r["id"] for r in rows]
        assert ids == [f"m{i}" for i in range(500)], f"{arm}: rows not the 500 problems in order"
        ok[arm] = np.array([bool(r["ok_rescored"]) for r in rows])
        fin = np.array([r["finish"] == "stop" for r in rows])
        toks = np.array([r["n_tokens"] for r in rows])
        cut = ~fin & ~np.array([r["closed"] for r in rows])  # stopped inside unfinished thinking
        g2 = [second_grade(r["gold"], texts[r["id"]]) for r in rows]
        gradable = np.array([g is not None for g in g2])
        ok2[arm] = np.array([bool(g) for g in g2])
        out[arm] = dict(
            run=name, accuracy=pct(ok[arm]), finished=pct(fin),
            acc_finished=pct(ok[arm][fin]), acc_cutoff=pct(ok[arm][cut]),
            points_from_cutoff=round(100 * int(ok[arm][cut].sum()) / len(rows), 1),
            mean_tokens_finished=round(float(toks[fin].mean())) if fin.any() else None,
            grader2_accuracy=pct(ok2[arm]), grader2_ungradable=int((~gradable).sum()),
            grader_agreement=pct(ok[arm][gradable] == ok2[arm][gradable]),
        )
        print(arm, json.dumps(out[arm]), flush=True)

    rng = np.random.default_rng(0)
    out["pairs"] = {f"{a}_vs_{b}": paired(ok[a], ok[b], rng) for a, b in PAIRS}
    rng = np.random.default_rng(0)
    out["pairs_grader2"] = {f"{a}_vs_{b}": paired(ok2[a], ok2[b], rng) for a, b in PAIRS}
    e1, o = out["E1"], out["O"]
    out["predictions"] = dict(
        A1_E1_finished_acc_at_least_10_below_O=e1["acc_finished"] <= o["acc_finished"] - 10,
        A2_E1_finished_shorter_than_O=e1["mean_tokens_finished"] < o["mean_tokens_finished"],
        A3_E1_cutoff_points_at_least_2=e1["points_from_cutoff"] >= 2.0,
    )
    out["both_graders_required"] = any(out[a]["grader_agreement"] < 97 for a in ARMS)
    for k in ("pairs", "pairs_grader2", "predictions", "both_graders_required"):
        print(k, json.dumps(out[k]))
    (Path(__file__).parent / "part_a.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
