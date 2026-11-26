"""Phase-15 endpoints: the endpoint script of phase 14, applied to this phase's runs, plus the
exploratory comparisons this phase adds (against the phase-14 editor's final and four other versions).

    python phase15/compute_endpoints.py     # prints the endpoints and rewrites phase15/endpoints.json

Needs the grader's environment (numpy, scipy, math-verify). Every rule, threshold and word is
phase14/endpoints.py's and is not restated here. This file names the runs of this phase, checks that
every re-scored file it reads is one reading (the tree's checker, one way of reading the thinking, and
figures that each run file and its re-score agree on), adds the exploratory comparisons, which carry
no protocol word, and writes the output once, into this folder.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase14"))
import endpoints as base  # noqa: E402  (the endpoint script of phase 14; its tests are phase14/test_endpoints.py)

PREVIOUS = "p14_final"   # the phase-14 editor's final: the exploratory comparison
# the seven runs of this report, by name: `main()` renames the base module's runs only while it runs
RUNS = ("p15_final", base.STOCK, base.ORIG, "p15_final_ref", base.REF_STOCK, base.REF_ORIG, PREVIOUS)
# other versions of the model that the record compares the final candidate with (exploratory; no protocol word)
OTHERS = (("p4_mimo", "the best blind quick fix (MiMo-flash, phase 4b)"),
          ("p10_fullbudget", "the 50-pass edit (phase 10)"),
          ("p7_stack_full_v2", "the published edit with our repair at full strength (phase 7)"),
          ("p7_uperp_on_loop", "our map repair (phase 7)"))


def checked(stem):
    """The re-scored file of `stem`, after three checks the phase-14 script does not make: the tree's
    checker stamp (the phase-14 script checks its own six runs; this covers every run read here), one way of
    reading the thinking in every file (`no_think`), and a re-score whose figures are those of the run
    file beside it (a run generated again after its re-score would otherwise pass unnoticed)."""
    try:
        rs, rj = base.rescored(stem), base.run(stem)
    except FileNotFoundError as e:
        missing = Path(e.filename).name
        sys.exit(f"{stem}: {e.strerror}: {missing} (re-score the run first)" if missing.endswith("_rescored.json")
                 else f"{stem}: {e.strerror}: {missing} (the run's files are not collected)")
    s, r = rs["summary"], rj["summary"]
    if s.get("rescored_with") != base.rescore.RESCORED_WITH:
        sys.exit(f"{stem} was re-scored with {s.get('rescored_with')}, not with the checker in this tree "
                 f"({base.rescore.RESCORED_WITH}): re-score it first")
    if bool(s.get("no_think")) != bool(base.rescored(RUNS[0])["summary"].get("no_think")):
        sys.exit(f"{stem} was re-scored with no_think={s.get('no_think')}, the final candidate with "
                 f"no_think={base.rescored(RUNS[0])['summary'].get('no_think')}: not one reading")
    for key in ("mean_tokens", "cap_hit_rate"):
        if s.get(key) != r.get(key):
            sys.exit(f"{stem}: its re-score reads {key} {s.get(key)} and its run file {r.get(key)}: "
                     "the run was generated again after it was re-scored")
    return rs


def verdicts(stem):
    return {r["id"]: bool(r["ok_rescored"]) for r in base.rescored(stem)["rows"]}


def main():
    saved = base.FINAL, base.REF_FINAL, base.HERE
    try:
        base.FINAL, base.REF_FINAL, base.HERE = "p15_final", "p15_final_ref", HERE   # stock, the published edit and the two reference arms keep their names
        for stem in RUNS + tuple(stem for stem, _ in OTHERS):
            checked(stem)
        out = base.build()                                   # the phase-14 checks and endpoints; nothing written
        out["exploratory_runs"] = {"phase-14 editor's final": base.describe(PREVIOUS, verdicts(PREVIOUS))}
        out["exploratory_pairs"] = [base.pair(verdicts(base.FINAL), verdicts(PREVIOUS),
                                              "final candidate against the phase-14 editor's final (no protocol word)")]
        out["other_exploratory_runs"] = {name: base.describe(stem, verdicts(stem)) for stem, name in OTHERS}
        out["other_exploratory_pairs"] = [base.pair(verdicts(base.FINAL), verdicts(stem), f"final candidate against {name} (no protocol word)")
                                          for stem, name in OTHERS]
        out["about"] += (" exploratory_runs, exploratory_pairs, other_exploratory_runs and other_exploratory_pairs are this phase's "
                         "additions and carry no protocol word.")
        text = json.dumps(out, indent=1, ensure_ascii=False)
        base.write_atomic(HERE / "endpoints.json", text + "\n")   # on disk first, so the console never shows more
        print(text)
    finally:
        base.FINAL, base.REF_FINAL, base.HERE = saved       # the module is shared with the phase-14 tests


if __name__ == "__main__":
    main()
