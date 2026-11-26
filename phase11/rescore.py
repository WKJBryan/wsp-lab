"""Re-score the phase-11 runs in place with phase3c/rescore.py (ISSUES #19).

Each run JSON's pred / ok / rule, its summary's accuracy / answered, and the run's validity
record are recomputed from the saved generations with the repository extractor, used with
no_think=True (Gemma has no think blocks): the checker's verdict (`rescore.judge`: a box, else
the marked line or a concluding last line, whichever is right). A last line that cannot be a
conclusion is not read, so `pred` is None there and "answered" counts it as unanswered.
Nothing else in the file changes.

Usage: uv run python phase11/rescore.py [<run> ...]   (no arguments = every top-level run with rows)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
from rescore import RESCORED_WITH, judge, write_json  # noqa: E402
from runcheck import check_run  # noqa: E402

TEXTS_OF = {"a1_wound_practice_8k": "a1_wound_practice"}   # the result file was renamed; its generations were not


def rescore_run(name):
    path = HERE / f"{name}.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    texts = json.loads((HERE / f"{TEXTS_OF.get(name, name)}_texts.json").read_text(encoding="utf-8"))
    for r in run["rows"]:
        r["ok"], r["pred"], r["rule"] = judge(r["gold"], texts[r["id"]], no_think=True)
    n = len(run["rows"])
    # the fragment rate is reported, not judged: a badly damaged model may produce such tails itself
    problems, validity = check_run(run["rows"], texts, ok_key="ok", soup_bar=None)
    run["summary"].update(
        accuracy=round(100 * sum(r["ok"] for r in run["rows"]) / n, 1),
        answered=round(100 * sum(r["pred"] is not None for r in run["rows"]) / n, 1),
        rescored_with=RESCORED_WITH, validity=dict(validity, problems=problems))
    write_json(path, run)
    return run["summary"]["accuracy"], n, problems


if __name__ == "__main__":
    if sys.argv[1:]:
        names = sys.argv[1:]
    else:   # every top-level result file (it has rows), skipping the small tables that have none
        names = [p.stem for p in sorted(HERE.glob("*.json"))
                 if "rows" in json.loads(p.read_text(encoding="utf-8"))]
    failed_any = False
    for name in names:
        acc, n, problems = rescore_run(name)
        print(json.dumps(dict(name=name, n=n, accuracy=acc, validity_problems=problems)))
        failed_any |= bool(problems)
    sys.exit(2 if failed_any else 0)
