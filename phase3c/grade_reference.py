"""Grade a run the way the original authors' evaluation did: the last \\boxed{} answer of each
generation, checked with math-verify. Reports the repository's own matcher on the same boxed
answers too, so the two graders are compared on identical extractions. For the calibration runs
(runners with --reference-question). No GPU.

Usage (the grader environment, which has math-verify):
  python phase3c/grade_reference.py <run.json> [<run.json> ...]
"""

import json
import sys
from pathlib import Path

from math_verify import parse, verify
from math_verify.errors import TimeoutException  # not an Exception subclass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase3b"))
from runcheck import load_texts  # noqa: E402
from rescore import last_boxed  # noqa: E402
from scorer import answers_match  # noqa: E402


def grade(run_path):
    run = json.loads(run_path.read_text(encoding="utf-8"))
    rows, texts = run["rows"], load_texts(run_path)
    boxed = {r["id"]: last_boxed(texts[r["id"]]) for r in rows}
    ok_matcher = ok_verify = ungraded = 0
    for r in rows:
        b = boxed[r["id"]]
        ok_matcher += b is not None and answers_match(r["gold"], b)
        if b is None:
            continue
        try:
            gold = parse(f"${r['gold']}$", raise_on_error=True)
            if not gold:
                ungraded += 1
            elif verify(gold, parse(f"${b}$", raise_on_error=True), raise_on_error=True):
                ok_verify += 1
        except (Exception, TimeoutException):
            ungraded += 1
    n = len(rows)
    boxed_rows = sum(b is not None for b in boxed.values())
    graded = boxed_rows - ungraded
    return dict(run=run_path.stem, n=n, boxed_pct=round(100 * boxed_rows / n, 1),
                accuracy_matcher=round(100 * ok_matcher / n, 1),
                # a row math-verify could not grade counts as wrong; among no graded rows there is no rate
                accuracy_math_verify=round(100 * ok_verify / n, 1) if graded else None,
                accuracy_math_verify_gradable=round(100 * ok_verify / graded, 1) if graded else None,
                ungraded=ungraded,
                warning=("math-verify could not grade more than 1% of rows" if ungraded > n / 100 else None),
                cap_hit_rate=run["summary"].get("cap_hit_rate"),
                finished_pct=run["summary"].get("finished"),
                mean_tokens=run["summary"].get("mean_tokens"),
                validity_problems=(run["summary"].get("validity") or {}).get("problems") or [])


if __name__ == "__main__":
    bad = False
    for arg in sys.argv[1:]:
        result = grade(Path(arg))
        print(json.dumps(result))
        bad |= bool(result["warning"]) or result["accuracy_math_verify"] is None or bool(result["validity_problems"])
    sys.exit(1 if bad else 0)     # a run that could not be graded, or failed its validity check, is no measurement
