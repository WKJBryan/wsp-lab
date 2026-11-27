"""Endpoints of a phase-14-style experiment: the endpoint script of phase 14 applied to the runs named on
the command line, plus exploratory comparisons that carry no protocol word.

    python phase16/compute_endpoints.py --final p16_A_final --out phase16/endpoints_A.json \
        [--ref-final p16_A_final_ref] \
        [--compare "NAME=STEM"] [--compare "NAME=STEM"] ... \
        [--other "NAME=STEM"] [--other "NAME=STEM"] ... \
        [--hint-versus STEM] [--hint-versus STEM] ...      (the told session only)

Each --compare, --other and --hint-versus flag takes ONE entry; repeat the flag for further runs.

Needs the grader's environment (numpy, scipy, math-verify). Every rule, threshold and word is
phase14/endpoints.py's and is not restated here. Stock, the published edit and the two reference arms
are those of phase 14. `--final` names this experiment's scored run under the repository's question and
`--ref-final` its run under the reference question (default: the final's name with `_ref`). Each
`--compare` and `--other` entry is `NAME=STEM`, a description and the stem of a re-scored run under
`phase3c/`; `--compare` holds the comparisons the phase's protocol registers and `--other` further
versions of the model the record mentions. All of them are paired with the final candidate and carry no
protocol word.

Before anything is built, every re-scored file read here is checked: the tree's checker stamp, one way of
reading the thinking (`no_think`), and figures that each run file and its re-score agree on (a run
generated again after its re-score would otherwise pass unnoticed). The output is written once, to the
file given with `--out`, which must be in this script's own folder, so that it can never overwrite
another phase's report.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase14"))
import endpoints as base  # noqa: E402  (the endpoint script of phase 14; its tests are phase14/test_endpoints.py)

FIXED = (base.STOCK, base.ORIG, base.REF_STOCK, base.REF_ORIG)   # stock, the published edit, the two reference arms


def parse_entries(groups):
    """[(stem, name)] from the NAME=STEM entries of several options, in the order given. A stem or a name
    may appear once only, across all the options: a second entry would be described and paired twice."""
    out = []
    for values in groups:
        for v in values or []:
            name, sep, stem = v.rpartition("=")
            if not sep or not name.strip() or not stem.strip():
                sys.exit(f"--compare and --other take NAME=STEM; got {v!r}")
            out.append((stem.strip(), name.strip()))
    for what, i in (("a run", 0), ("a name", 1)):
        seen = [e[i] for e in out]
        twice = sorted({s for s in seen if seen.count(s) > 1})
        if twice:
            sys.exit(f"{what} is named twice: {twice}")
    return out


def files(stem):
    """(re-score, run) of `stem`; a missing file stops with a message naming the run and the file."""
    try:
        return base.rescored(stem), base.run(stem)
    except FileNotFoundError as e:
        missing = Path(e.filename).name
        sys.exit(f"{stem}: {e.strerror}: {missing} (re-score the run first)" if missing.endswith("_rescored.json")
                 else f"{stem}: {e.strerror}: {missing} (the run's files are not collected)")


def checked(stem, reference):
    """The re-scored file of `stem`, after three checks the phase-14 script does not make: the tree's
    checker stamp, the same way of reading the thinking as the final candidate (`reference`), and a
    re-score whose figures are those of the run file beside it."""
    rs, rj = files(stem)
    s, r = rs["summary"], rj["summary"]
    if s.get("rescored_with") != base.rescore.RESCORED_WITH:
        sys.exit(f"{stem} was re-scored with {s.get('rescored_with')}, not with the checker in this tree "
                 f"({base.rescore.RESCORED_WITH}): re-score it first")
    if bool(s.get("no_think")) != bool(reference):
        sys.exit(f"{stem} was re-scored with no_think={s.get('no_think')}, the final candidate with no_think={bool(reference)}: not one reading")
    for key in ("mean_tokens", "cap_hit_rate"):
        if s.get(key) != r.get(key):
            sys.exit(f"{stem}: its re-score reads {key} {s.get(key)} and its run file {r.get(key)}: "
                     "the run was generated again after it was re-scored")
    return rs


def verdicts(stem):
    return {r["id"]: bool(r["ok_rescored"]) for r in base.rescored(stem)["rows"]}


def check_kinds(final, ref_final, entries):
    """The runs must be of the right kinds, so that a wrong name on the command line cannot pair the wrong
    runs. The final and its reference run must be the same model under the two questions, and every
    compared run must have been run under the repository's question. Each is taken from the run file's
    `settings`; a run whose file records none (the oldest runs) is not checked for kind."""
    if ref_final == final:
        sys.exit(f"--ref-final names the final's own run ({final}): the reference run is another run")
    for stem in (s for s, _ in entries):
        if stem in (final, ref_final) + FIXED:
            sys.exit(f"{stem} is compared with itself: it is the final, its reference run, or one of the fixed runs")
    s_final, s_ref = files(final)[1]["summary"].get("settings") or {}, files(ref_final)[1]["summary"].get("settings") or {}
    if s_final and s_ref and s_final.get("model") != s_ref.get("model"):
        sys.exit(f"{final} and {ref_final} were not runs of the same model ({s_final.get('model')} and {s_ref.get('model')})")
    if s_final and s_final.get("reference_question") is not False:
        sys.exit(f"{final} was not run under the repository's question (reference_question={s_final.get('reference_question')})")
    if s_ref and s_ref.get("reference_question") is not True:
        sys.exit(f"{ref_final} was not run under the reference question (reference_question={s_ref.get('reference_question')})")
    for stem, _ in entries:
        s = files(stem)[1]["summary"].get("settings") or {}
        if s and s.get("reference_question") is not False:
            sys.exit(f"{stem} was run under the reference question, so it is not comparable with the final under the repository's question")


HINT_RULE = ("told final against each blind final, told minus blind: THE HINT HELPS if more than 4.0 points above every "
             "blind final with every 95% interval excluding 0; THE HINT HURTS if more than 4.0 points below every blind "
             "final with every interval excluding 0; NO MEASURABLE EFFECT if within 4.0 points of every blind final; "
             "otherwise INCONCLUSIVE")


def hint_word(pairs):
    """The registered rule (phase16/protocol.md) applied to the told final's pairs with the blind finals."""
    if not pairs:
        return "INCONCLUSIVE"
    d = [p["delta"] for p in pairs]
    lo = [p["ci95"][0] for p in pairs]
    hi = [p["ci95"][1] for p in pairs]
    if all(x > 4.0 and l > 0 for x, l in zip(d, lo)):
        return "THE HINT HELPS"
    if all(x < -4.0 and h < 0 for x, h in zip(d, hi)):
        return "THE HINT HURTS"
    if all(abs(x) <= 4.0 for x in d):
        return "NO MEASURABLE EFFECT"
    return "INCONCLUSIVE"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Endpoints of a phase-14-style experiment (see the module docstring).")
    ap.add_argument("--final", required=True, help="stem of this experiment's scored run under the repository's question")
    ap.add_argument("--ref-final", help="stem of its run under the reference question (default: FINAL_ref)")
    ap.add_argument("--out", required=True, help="the output file, in this script's own folder")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=STEM",
                    help="a comparison the protocol registers (repeatable; a stem or a name may appear once across both options)")
    ap.add_argument("--other", action="append", default=[], metavar="NAME=STEM", help="a further version of the model (repeatable)")
    ap.add_argument("--hint-versus", action="append", default=[], metavar="STEM",
                    help="the told session only: the blind finals it is judged against by the phase-16 rule (repeatable; each must also be a --compare stem)")
    a = ap.parse_args(argv)
    a.final = a.final.strip()
    if not a.final:
        sys.exit("--final must name a run")
    ref_final = (a.ref_final or "").strip() or f"{a.final}_ref"
    entries = parse_entries([a.compare, a.other])
    compare, other = entries[:len(a.compare)], entries[len(a.compare):]
    out_path = Path(a.out.strip())
    if not out_path.is_absolute():
        out_path = HERE.parent / out_path
    if out_path.resolve().parent != HERE.resolve():
        sys.exit(f"--out must be a file in this script's own folder, relative to the repository root (such as "
                 f"phase16/endpoints_A.json), so that it can never overwrite another phase's report; got {a.out}")
    hint = [s.strip() for s in a.hint_versus]
    if len(set(hint)) != len(hint):
        sys.exit(f"a run is named twice under --hint-versus: {sorted(s for s in hint if hint.count(s) > 1)}")
    missing = [s for s in hint if s not in {stem for stem, _ in compare}]
    if missing:
        sys.exit(f"--hint-versus names runs that are not --compare entries: {missing}")
    fixed_stems = (a.final, ref_final) + FIXED + tuple(s for s, _ in compare) + tuple(s for s, _ in other)
    saved = base.FINAL, base.REF_FINAL, base.HERE
    try:
        base.FINAL, base.REF_FINAL, base.HERE = a.final, ref_final, out_path.parent
        reference = bool(files(a.final)[0]["summary"].get("no_think"))   # the final candidate's way of reading the thinking
        check_kinds(a.final, ref_final, compare + other)
        for stem in fixed_stems:
            checked(stem, reference)
        out = base.build()                                   # the phase-14 checks and endpoints; nothing written
        for key, entries in (("exploratory", compare), ("other_exploratory", other)):
            out[f"{key}_runs"] = {name: base.describe(stem, verdicts(stem)) for stem, name in entries}
            out[f"{key}_pairs"] = [dict(base.pair(verdicts(a.final), verdicts(stem), f"final candidate against {name}"), stem=stem)
                                   for stem, name in entries]
        if hint:
            pairs = {p["stem"]: p for p in out["exploratory_pairs"]}
            out["hint"] = dict(rule=HINT_RULE, against={s: hint_word([pairs[s]]) for s in hint}, word=hint_word([pairs[s] for s in hint]))
        out["about"] += (f" final: {a.final}; reference run: {ref_final}. exploratory_runs, exploratory_pairs, "
                         "other_exploratory_runs and other_exploratory_pairs are comparisons without a protocol word; "
                         "each pair also names the run it compares under `stem`."
                         + (" `hint` applies the phase-16 rule for the told session to its pairs with the blind finals." if hint else ""))
        text = json.dumps(out, indent=1, ensure_ascii=False)
        base.write_atomic(out_path, text + "\n")             # on disk first, so the console never shows more
        print(text)
    finally:
        base.FINAL, base.REF_FINAL, base.HERE = saved       # the module is shared with the phase-14 tests


if __name__ == "__main__":
    main()
