"""Phase-14 endpoints from the collected files. No GPU.

    python phase14/endpoints.py            # prints the endpoints and rewrites phase14/endpoints.json

Needs the grader's environment (numpy, math-verify). The module itself imports with the standard
library only; the heavy imports are made where they are used.

Primary endpoint (phase14/protocol.md): MATH-500 under the repository's question, the final answer
checker, paired against stock and the published edit. Secondary endpoint: the reference question,
the last boxed answer of each generation, both graders on the same boxed answers, paired against
the two reference arms. Paired statistics as in phase 8 (10,000-draw bootstrap, exact McNemar test).
Also printed: the primary comparison under the two other ways of choosing the answer line
(ISSUES #31). The protocol's ±4.0-point words are defined for the primary comparison with the
published edit only; no other comparison gets one. For stock the protocol's word is REPAIR (at
least 16.0% and at least stock + 10.0 points) or not, decided here on counts, not on rounded
percentages.
"""

import gzip
import importlib.metadata
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3c"))
sys.path.insert(0, str(HERE.parent / "phase3b"))
import rescore  # noqa: E402  (the final answer checker; its line rule is re-run under two variants)
from scorer import answers_match  # noqa: E402

P3 = HERE.parent / "phase3c"
FINAL, STOCK, ORIG = "p14_final", "p3c_stock", "p3c_c3_original"
REF_FINAL, REF_STOCK, REF_ORIG = "p14_final_ref", "ref_stock", "ref_orig"
PROBLEMS = 500
REPAIR_MIN_PERCENT, REPAIR_MIN_GAP_POINTS = 16.0, 10.0
ADMISSION_FLOOR = 100   # runs of this many rows or more carry the admission-order check (phase3c/runcheck.py)
# the checker's own boxed extractor: one brace level
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
# reading C of ISSUES #31, exactly as it was measured there: the shape of a concluding line is a
# $math$ span, a display block, a bold value, or a linking word and a number. The word boundary
# before the linking word also stands before "=", so a spaced equals sign is not recognised.
STRUCT = re.compile(r"\$\$?[^$]+\$|\\\[|\\\(|\*\*[^*]+\*\*|\b(?:is|are|equals|=)\s*:?\s*\**\s*-?\d",
                    re.IGNORECASE)


def word(d, lo, hi):
    """The protocol's word for the primary comparison with the published edit."""
    if abs(d) <= 4.0:
        return "REACHES IT"
    if d < -4.0 and hi < 0:
        return "FALLS SHORT"
    if d > 4.0 and lo > 0:
        return "EXCEEDS"
    return "INCONCLUSIVE"


def repaired(right_final, right_stock, n):
    """REPAIR or not, on counts: at least REPAIR_MIN_PERCENT, and at least REPAIR_MIN_GAP_POINTS
    above stock. Products of these constants with whole counts are exact in floating point, so a
    result exactly on a bar is decided as the protocol says (a difference of rounded percentages
    would not be). `min_right` and `min_gap` are the smallest whole counts that clear each bar."""
    holds = n > 0 and 100 * right_final >= REPAIR_MIN_PERCENT * n and 100 * (right_final - right_stock) >= REPAIR_MIN_GAP_POINTS * n
    return dict(rule=f"at least {REPAIR_MIN_PERCENT}% and at least stock + {REPAIR_MIN_GAP_POINTS} points",
                right_final=right_final, right_stock=right_stock, n=n,
                min_right=math.ceil(REPAIR_MIN_PERCENT * n / 100), min_gap=math.ceil(REPAIR_MIN_GAP_POINTS * n / 100),
                holds=holds)


def last_box(text):
    """The content of the last one-level \\boxed{...} of a generation; "" when there is none."""
    found = BOXED.findall(text)
    return found[-1] if found else ""


def readings(text, gold, ok):
    """(the checker's verdict, the section's last line read instead, reading C) for one generation,
    with the checker's own cleanup and matcher. A generation with a boxed answer, or without an
    answer section, reads the same under every rule; an answer section with no text reads wrong."""
    if BOXED.search(text) or "</think>" not in text:
        return (ok, ok, ok)
    section = text.split("</think>", 1)[1]
    lines = [l.strip().replace("</think>", " ").strip() for l in section.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return (ok, False, ok)
    last = lines[-1]
    cand = rescore.clean(last)
    last_ok = bool(cand) and answers_match(gold, cand)
    return (ok, last_ok, last_ok if STRUCT.search(last) else ok)


def math_verify_ok(gold, boxed):
    """True or False under math-verify; None when math-verify cannot read the answer (which the
    grader counts as ungraded, not as wrong: `phase3c/grade_reference.py`)."""
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException   # a BaseException subclass (ISSUES #17)
    try:
        g = parse(f"${gold}$", raise_on_error=True)
        return bool(g) and bool(verify(g, parse(f"${boxed}$", raise_on_error=True), raise_on_error=True))
    except (Exception, TimeoutException):
        return None


def rescored(stem):
    return json.loads((P3 / f"{stem}_rescored.json").read_text(encoding="utf-8"))


def run(stem):
    return json.loads((P3 / f"{stem}.json").read_text(encoding="utf-8"))


def texts(stem):
    with gzip.open(P3 / f"{stem}_texts.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def describe(stem, right):
    """A run's level: accuracy from the verdicts given, the rest from the run and its re-score.
    A run that failed a validity check, or has none on record, is not a measurement (ISSUES #20):
    the record must show no problem, a judged fragment-tail check and, for a run of
    ADMISSION_FLOOR rows or more, a run admission-order check."""
    rs, rj = rescored(stem), run(stem)
    rows, sm = rs["rows"], rj["summary"]
    validity = rs["summary"].get("validity")
    sound = (isinstance(validity, dict) and validity.get("problems") == [] and validity.get("soup_judged") is True
             and (len(rows) < ADMISSION_FLOOR or validity.get("admission_tested") is True))
    if not sound:
        sys.exit(f"{stem} is not a measurement: its validity record reads {validity} (ISSUES #20)")
    return dict(accuracy=round(100 * sum(right.values()) / len(right), 1), right=sum(right.values()),
                n=len(right), cap_hit_rate=sm.get("cap_hit_rate"),
                closed=round(100 * sum(bool(r["closed"]) for r in rows) / len(rows), 1),
                mean_tokens=sm.get("mean_tokens"), wall_s=sm.get("wall_s"), validity=validity,
                rescore_file_stamp=rs["summary"].get("rescored_with"))


def pair(xa, ya, label, word_key=None):
    """Paired statistics over the problems both runs hold, which must be all of them.
    `word_key` names the field the protocol's word goes under; None where the protocol has none."""
    if len(xa) != PROBLEMS or len(ya) != PROBLEMS or set(xa) != set(ya):
        sys.exit(f"{label}: the two runs do not hold the same {PROBLEMS} problems")
    import numpy as np
    sys.path.insert(0, str(HERE.parent / "phase8"))
    from analyze_saved import paired
    ids = sorted(xa)
    x = np.array([xa[i] for i in ids])
    y = np.array([ya[i] for i in ids])
    p = paired(x, y, np.random.default_rng(0))
    lo, hi = float(p["ci95"][0]), float(p["ci95"][1])
    out = dict(label=label, n=len(ids), first=round(100 * float(x.mean()), 1),
               second=round(100 * float(y.mean()), 1), delta=p["delta"], ci95=[round(lo, 1), round(hi, 1)],
               only_first=p["only_first"], only_second=p["only_second"],
               p=float(f"{p['mcnemar_p']:.3g}"))
    if word_key:
        out[word_key] = word(p["delta"], lo, hi)
    return out


def line_readings(stem):
    """{problem id: readings} for a run under the repository's question."""
    t, out = texts(stem), {}
    for r in rescored(stem)["rows"]:
        out[r["id"]] = readings(t[r["id"]], r["gold"], bool(r["ok_rescored"]))
    return out


def boxed_readings(stem):
    """({problem id: (the repository's matcher on the last boxed answer, math-verify on it, or
    None when math-verify cannot read it)}, rows with a boxed answer, rows math-verify cannot read)."""
    t, out, attempted, unreadable = texts(stem), {}, 0, 0
    for r in rescored(stem)["rows"]:
        boxed = last_box(t[r["id"]])
        mv = False
        if boxed:
            attempted += 1
            mv = math_verify_ok(r["gold"], boxed)
            unreadable += mv is None
        out[r["id"]] = (bool(boxed) and answers_match(r["gold"], boxed), mv)
    return out, attempted, unreadable


def write_atomic(path, text):
    """Replace a file only once the new content is complete on disk; line endings as given."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def grader_versions():
    try:
        mv = "math-verify " + importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        mv = "math-verify (version not recorded)"
    return dict(matcher="phase3b/scorer.py answers_match on the last boxed answer",
                math_verify=mv, problems_in_sorted_order="the bootstrap resamples the problems in sorted id order")


def main():
    out = {"about": "rescore_file_stamp identifies the checker that produced each re-score file; in secondary_runs "
                    "'accuracy' is the repository's matcher on the last boxed answer, computed here, and "
                    "'accuracy_math_verify' is the second grader."}
    checker = {s: {r["id"]: bool(r["ok_rescored"]) for r in rescored(s)["rows"]} for s in (FINAL, STOCK, ORIG)}
    out["primary_runs"] = {"final candidate": describe(FINAL, checker[FINAL]), "stock": describe(STOCK, checker[STOCK]),
                           "published edit": describe(ORIG, checker[ORIG])}
    out["primary_pairs"] = [pair(checker[FINAL], checker[STOCK], "final candidate against stock"),
                            pair(checker[FINAL], checker[ORIG], "final candidate against the published edit", "word")]
    out["repair"] = repaired(sum(checker[FINAL].values()), sum(checker[STOCK].values()), len(checker[FINAL]))

    lines = {s: line_readings(s) for s in (FINAL, STOCK, ORIG)}
    out["line_rules"] = []
    for c, rule in enumerate("ABC"):
        for other, name in ((STOCK, "stock"), (ORIG, "the published edit")):
            out["line_rules"].append(dict(rule=rule, **pair({i: v[c] for i, v in lines[FINAL].items()},
                                                            {i: v[c] for i, v in lines[other].items()},
                                                            f"final candidate against {name}",
                                                            "word_if_read_this_way" if other == ORIG else None)))

    boxed = {}
    out["secondary_runs"], out["graders"] = {}, grader_versions()
    for s in (REF_FINAL, REF_STOCK, REF_ORIG):
        boxed[s], attempted, unreadable = boxed_readings(s)
        if attempted and (unreadable > PROBLEMS // 100 or unreadable == attempted):
            sys.exit(f"{s}: math-verify could not read {unreadable} of {attempted} boxed answers")
        info = describe(s, {i: x[0] for i, x in boxed[s].items()})
        info["accuracy_math_verify"] = round(100 * sum(x[1] is True for x in boxed[s].values()) / len(boxed[s]), 1)
        info["boxed_answers"] = attempted
        info["math_verify_unreadable"] = unreadable
        out["secondary_runs"][s] = info
    out["secondary_pairs"] = []
    for other in (REF_STOCK, REF_ORIG):
        for g, name in ((0, "repository matcher"), (1, "math-verify")):
            # only True counts under math-verify: an answer it cannot read (None) is not right
            mine = {i: x[g] is True for i, x in boxed[REF_FINAL].items()}
            theirs = {i: x[g] is True for i, x in boxed[other].items()}
            out["secondary_pairs"].append(pair(mine, theirs, f"final candidate against {other}, {name}"))

    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    write_atomic(HERE / "endpoints.json", text + "\n")


if __name__ == "__main__":
    main()
