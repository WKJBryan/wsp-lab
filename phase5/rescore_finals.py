"""Re-score the phase-5 finals with phase3c/rescore.py (ISSUES #19).

The two v2 finals hold four samples per problem (final_vllm2.py, sequential n=1 passes). They were
scored at run time by final_vllm2.py's own extractor, which is not the repository's; this re-scores
every saved generation with phase3c/rescore.py and compares the two arms, paired over the 2,000
problem x sample pairs. Writes <name>_rescored.json next to each final.

Usage: uv run python phase5/rescore_finals.py
"""

import gzip
import json
import random
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
from rescore import RESCORED_WITH, answers_match, extract_answer, write_json  # noqa: E402
from runcheck import check_run  # noqa: E402

ARMS = ("final5v2_start", "final5v2_dose175")     # published edit at 1.0x, and at 1.75x


def rescore(name):
    run = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    with gzip.open(HERE / f"{name}_texts.json.gz", "rt", encoding="utf-8") as f:
        texts = json.load(f)
    samples = sorted({k.split("#")[1] for k in texts})
    # the rows carry no id: they are the 500 scored problems in order (asserted against the gold answers)
    golds = json.loads((HERE.parent / "phase3c" / "problems.json").read_text(encoding="utf-8"))
    assert [r["gold"] for r in run["rows"]] == [p["gold"] for p in golds], f"{name}: rows are not m0..m499 in order"
    assert len(texts) == len(run["rows"]) * len(samples), f"{name}: generations do not pair with rows"
    rows, ok = [], {}
    for i, r in enumerate(run["rows"]):
        preds = [extract_answer(texts[f"m{i}#{s}"]) for s in samples]
        oks = [answers_match(r["gold"], p) for p in preds]
        ok[f"m{i}"] = oks
        rows.append(dict(id=f"m{i}", gold=r["gold"], pred_rescored=preds, ok_rescored=oks))
    n = len(rows) * len(samples)
    per_sample = [round(100 * sum(v[k] for v in ok.values()) / len(rows), 1) for k in range(len(samples))]
    # one validity record per sample: a sample's generations are in submission order like any run's
    # (the rows carry no finish or closed field, so the outcome tested by position is "correct")
    validity, problems = {}, []
    for k, s in enumerate(samples):
        sample_rows = [dict(id=f"m{i}", ok_rescored=ok[f"m{i}"][k]) for i in range(len(rows))]
        sample_texts = {f"m{i}": texts[f"m{i}#{s}"] for i in range(len(rows))}
        found, validity[s] = check_run(sample_rows, sample_texts, ok_key="ok_rescored")
        problems += [f"sample {s}: {p}" for p in found]
    summary = dict(name=f"{name}_rescored", n=len(rows), samples=len(samples), per_sample=per_sample,
                   accuracy=round(100 * sum(sum(v) for v in ok.values()) / n, 1),
                   accuracy_at_run_time=run["summary"]["accuracy_mean_over_generations"],
                   cap_hit_rate=run["summary"]["cap_hit_rate"], rescored_with=RESCORED_WITH,
                   validity=dict(per_sample=validity, problems=problems))
    write_json(HERE / f"{name}_rescored.json", dict(summary=summary, rows=rows))
    return summary, ok, problems


def paired(a, b, seed=20261124, draws=10000):
    """Difference a - b in points, 95% bootstrap interval over problems, exact McNemar p."""
    ids = sorted(a)
    x = [sum(int(p) and not q for p, q in zip(a[i], b[i])) for i in ids]   # per problem: a right, b wrong
    y = [sum(int(q) and not p for p, q in zip(a[i], b[i])) for i in ids]   # per problem: b right, a wrong
    k = len(next(iter(a.values())))
    n_pairs = len(ids) * k
    n_a, n_b = sum(map(sum, a.values())), sum(map(sum, b.values()))
    rng = random.Random(seed)
    diffs = sorted(100 * (sum(x[j] for j in picks) - sum(y[j] for j in picks)) / n_pairs
                   for picks in ([rng.randrange(len(ids)) for _ in ids] for _ in range(draws)))
    w, l = sum(x), sum(y)
    m = w + l
    tail = sum(comb(m, i) for i in range(0, min(w, l) + 1)) / 2 ** m
    return dict(a=round(100 * n_a / n_pairs, 1), b=round(100 * n_b / n_pairs, 1),
                difference=round(100 * (n_a - n_b) / n_pairs, 1),
                ci95=[round(diffs[int(0.025 * draws)], 1), round(diffs[int(0.975 * draws)], 1)],
                a_only=w, b_only=l, mcnemar_p=float(f"{min(1.0, 2 * tail):.3g}"))


if __name__ == "__main__":
    out = {name: rescore(name) for name in ARMS}
    for name, (summary, _, problems) in out.items():
        print(json.dumps(summary))
        if problems:
            print(f"RUN VALIDITY FAILED for {name} (ISSUES #20): " + "; ".join(problems), file=sys.stderr)
    print(json.dumps(dict(paired_start_minus_dose175=paired(out[ARMS[0]][1], out[ARMS[1]][1]))))
    sys.exit(2 if any(o[2] for o in out.values()) else 0)
