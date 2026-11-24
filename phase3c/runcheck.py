"""Validity checks for a finished generation run. A run that fails one is not a measurement.

Two signatures, both taken from the one corrupted final in this repository (ISSUES #20, the
phase-7 `stack_full` final):
  soup       the share of generations whose last 1500 characters contain characters of one of
             six non-Latin scripts (kana, CJK, Hangul, Cyrillic, Arabic, Devanagari).
             That run: 92%. Every other Qwen run: at most 0.4%. The damaged Gemma wound reaches
             31-50% on some of its editor passes, so the bar is 20% and a caller may report the
             rate without judging it.
  admission  the first half of the requests (rows are in submission order) against the second
             half, exact Fisher test, on three outcomes: finished, thinking closed, correct.
             In lockstep decoding an engine event hits requests by their position in the batch,
             which a property of the model cannot do. That run: 12.0% correct in the first half,
             0.8% in the second. Every other 500-problem run: p >= 0.2.
             Not detected: a contiguous outage away from the middle of the batch.

The checks are pure functions of a run's rows and texts; the metrics record whether each test
ran. The runner decides what to do.

Usage: uv run python phase3c/runcheck.py <run.json> [--soup-report-only]
Exit status 2 when a check fails.
"""

import gzip
import json
import math
import re
import sys
from pathlib import Path

SCRIPT = re.compile("[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0900-\u097f]")
SOUP_BAR = 20.0     # percent of generations
P_BAR = 0.001       # two-sided, per outcome
MIN_ROWS = 100      # below this the admission test has too few rows in each half to mean anything
TEXTS_OF = {"a1_wound_practice_8k": "a1_wound_practice"}   # a renamed result; its generations kept the old name


def soup_rate(texts, ids, tail=1500):
    """Percent of generations whose last `tail` characters contain a non-Latin script."""
    return 100.0 * sum(bool(SCRIPT.search(texts[i][-tail:])) for i in ids) / len(ids)


def fisher_p(a, b, c, d):
    """Two-sided exact test for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def p(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, r1 + c1 - n), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9)))


def admission_split(rows, outcome):
    """(rate first half %, rate second half %, two-sided Fisher p) for a boolean outcome."""
    half = len(rows) // 2
    a = sum(bool(outcome(r)) for r in rows[:half])
    c = sum(bool(outcome(r)) for r in rows[half:])
    return 100.0 * a / half, 100.0 * c / (len(rows) - half), fisher_p(a, half - a, c, len(rows) - half - c)


def check_run(rows, texts, ok_key=None, soup_bar=SOUP_BAR):
    """Return (problems, metrics). An empty list means every test that ran passed.

    A run with no rows, or with a row that has no saved generation, is reported as that one
    problem and nothing else is tested (soup_pct is then None). A run with fewer than MIN_ROWS
    rows is never given the admission test; its metrics say so (admission_tested false).
    soup_bar=None reports the fragment rate without judging it (the Gemma evaluator: a badly
    damaged model may produce such tails on its own).
    """
    metrics, problems = {"soup_pct": None, "soup_judged": soup_bar is not None, "admission_tested": False}, []
    if not rows:
        return ["rows: the run has no rows"], metrics
    ids = [r["id"] for r in rows]
    missing = [i for i in ids if i not in texts]
    if missing:
        return [f"generations: {len(missing)} of {len(ids)} rows have no saved generation"], metrics
    # the rate is compared as rounded to one decimal: at 500 rows it moves in steps of 0.2 points,
    # so a rate above the bar never rounds down to it
    metrics["soup_pct"] = round(soup_rate(texts, ids), 1)
    if soup_bar is not None and metrics["soup_pct"] > soup_bar:
        problems.append(f"soup: {metrics['soup_pct']}% of generations end in non-Latin script (bar {soup_bar}%)")
    metrics["admission_tested"] = False
    if len(rows) >= MIN_ROWS:
        outcomes = {}
        if any("finish" in r for r in rows):       # older Gemma runs never recorded why a text ended
            outcomes["finished"] = lambda r: r.get("finish") == "stop"
        if any(r.get("closed") is not None for r in rows):
            outcomes["closed"] = lambda r: bool(r.get("closed"))
        if ok_key and any(ok_key in r for r in rows):
            outcomes["correct"] = lambda r: bool(r[ok_key])
        metrics["admission_tested"] = bool(outcomes)   # false when no outcome is on record to test
        for name, f in outcomes.items():
            first, second, p = admission_split(rows, f)
            metrics[f"admission_{name}"] = [round(first, 1), round(second, 1), float(f"{p:.4g}")]
            if p < P_BAR:
                problems.append(f"admission order ({name}): first half {first:.1f}%, second half {second:.1f}%, "
                                f"Fisher p = {p:.2g} (bar {P_BAR})")
    return problems, metrics


def load_texts(run_path):
    """<run>_texts.json or <run>_texts.json.gz next to a result file."""
    run_path = Path(run_path)
    plain, zipped = run_path.with_name(run_path.stem + "_texts.json"), run_path.with_name(run_path.stem + "_texts.json.gz")
    if plain.exists():
        return json.loads(plain.read_text(encoding="utf-8"))
    if zipped.exists():
        with gzip.open(zipped, "rt", encoding="utf-8") as f:
            return json.load(f)
    sys.exit(f"no generations file next to {run_path.name}")


def save_run(result_path, result, texts_path, texts, compress=False):
    """Write a finished run so that a result file never exists without its generations and never
    replaces an older one: refuse if either file exists, write the generations first and the
    result last, each through a temporary file that replaces its target only once complete."""
    result_path, texts_path = Path(result_path), Path(texts_path)
    refuse_existing(result_path, texts_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)   # a new --outdir must not lose the run
    texts_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(texts, indent=1)   # non-ASCII stays escaped: older readers name no encoding
    tmp_texts = texts_path.with_name(texts_path.name + ".tmp")
    if compress:
        with gzip.open(tmp_texts, "wb") as f:       # bytes in, bytes out: the same file on every platform
            f.write(data.encode("utf-8"))
    else:
        tmp_texts.write_bytes(data.encode("utf-8"))
    tmp_texts.replace(texts_path)
    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_bytes(json.dumps(result, indent=1).encode("utf-8"))
    tmp.replace(result_path)


def refuse_existing(*paths):
    """Exit before anything is computed if any of these files already exists."""
    for p in paths:
        if Path(p).exists():
            sys.exit(f"refusing to overwrite {p}; choose a new name (ISSUES #20)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    run_path = Path(args[0])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows = run["rows"]
    if not rows:      # nothing to test: report it in the usual form, with the usual exit status
        print(json.dumps(dict(run=run_path.stem, soup_pct=None, soup_judged=True, admission_tested=False,
                              problems=["rows: the run has no rows"])))
        sys.exit(2)
    texts = load_texts(run_path.with_name(TEXTS_OF.get(run_path.stem, run_path.stem) + ".json"))
    ok_key = next((k for k in ("ok_rescored", "ok") if k in rows[0]), None)
    if ok_key is None:   # phase-3c style: the scored verdicts live in <run>_rescored.json
        rescored = run_path.with_name(run_path.stem + "_rescored.json")
        if rescored.exists():
            ok = {r["id"]: r["ok_rescored"] for r in json.loads(rescored.read_text(encoding="utf-8"))["rows"]}
            rows = [dict(r, ok_rescored=ok[r["id"]]) for r in rows]
            ok_key = "ok_rescored"
    problems, metrics = check_run(rows, texts, ok_key, soup_bar=None if "--soup-report-only" in sys.argv else SOUP_BAR)
    print(json.dumps(dict(run=run_path.stem, **metrics, problems=problems)))
    sys.exit(2 if problems else 0)


if __name__ == "__main__":
    main()
