"""Re-score the phase-8 arms in place with phase3c/rescore.py (ISSUES #19).

Each arm JSON's pred_rescored / ok_rescored / rule, its summary's accuracy / answered, and the
run's validity record are recomputed from the arm's saved generations with the checker's verdict
(`rescore.judge`: a box, else the marked line or a concluding last line, whichever is right).
Nothing else in the file changes.

Usage: uv run python phase8/rescore_arms.py [<arm> ...]      (no arguments = every arm)
"""

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
from rescore import RESCORED_WITH, judge, write_json  # noqa: E402
from runcheck import check_run  # noqa: E402


def rescore_arm(name):
    path = HERE / f"{name}.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(HERE / f"{name}_texts.json.gz", "rt", encoding="utf-8") as f:
        texts = json.load(f)
    no_think = bool(run["summary"]["settings"].get("no_think"))
    for r in run["rows"]:
        r["ok_rescored"], r["pred_rescored"], r["rule"] = judge(r["gold"], texts[r["id"]], no_think)
    n = len(run["rows"])
    problems, validity = check_run(run["rows"], texts, ok_key="ok_rescored")
    run["summary"].update(
        accuracy=round(100 * sum(r["ok_rescored"] for r in run["rows"]) / n, 1),
        answered=round(100 * sum(r["pred_rescored"] is not None for r in run["rows"]) / n, 1),
        rescored_with=RESCORED_WITH, validity=dict(validity, problems=problems))
    write_json(path, run)
    return run["summary"]["accuracy"], n, problems


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name[:-len("_texts.json.gz")] for p in HERE.glob("*_texts.json.gz"))
    failed_any = False
    for name in names:
        acc, n, problems = rescore_arm(name)
        print(json.dumps(dict(name=name, n=n, accuracy=acc, validity_problems=problems)))
        failed_any |= bool(problems)
    sys.exit(2 if failed_any else 0)
