"""Tests for phase16/compute_endpoints.py, which runs the endpoint script of phase 14 (phase14/endpoints.py, tested in
phase14/test_endpoints.py) under the run names given on the command line and writes, once, the endpoints file named
with --out, with the comparisons named with --compare and --other (the level of each compared run, and the pair of the
final candidate against it) and, for the told session, the word of the phase-16 rule over the pairs named with
--hint-versus.
  ParseEntriesTest  the NAME=STEM entries of --compare and --other;
  HintWordTest      the rule's word for the told final's pairs with the blind finals;
  ImportTest        importing the script: it runs nothing, and leaves the phase-14 script as that script made itself;
  MainTest          the arguments; the rule for --out; the kinds of the runs, and the checks before anything is built
                    (the tree's checker, one way of reading the thinking, figures that agree with the run file, files
                    that are there); the build under the command-line names; the comparisons; the hint; the one write
                    and the one print; the names put back; a failure, which leaves no output file.

No run file is read, and nothing is written into the repository. Stood in: the script's folder, which makes the
repository root of the rule for --out a temporary folder; the phase-14 script's folder; its readers rescored() and
run(), which answer from small files; its build(), which returns a small document; its main(); and its pair(), which
returns a fixed result. Real: the script's checked() (each call noted first), and the phase-14 script's describe()
and write_atomic(), the latter into the temporary folder (their calls noted). main() sets FINAL, REF_FINAL and HERE on
the phase-14 script and puts them back itself; the tests stand other values in first, and tearDown() puts every
stood-in attribute back as it was: the phase-14 script is one module object, shared with its own tests when the
suites run in one process.
Standard library only.
Usage (from the repository root):
  python -B -m unittest -v phase16/test_compute_endpoints.py
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_script():
    """phase16/compute_endpoints.py as a module object, imported from its file under a name of its own and so
    executed afresh. Not under "endpoints", the name it gives the phase-14 script: the phase-14 tests import that
    script under it, and all suites must get the one module object."""
    spec = importlib.util.spec_from_file_location("p16_compute_endpoints", HERE / "compute_endpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_script()
base = mod.base   # phase14/endpoints.py: the script that compute_endpoints.py wraps

FINAL = "p16_A_final"                 # this experiment's final candidate, as --final names it
REF_FINAL = "p16_A_final_ref"         # its reference run: the final's name with _ref, to which --ref-final defaults
OUT = "phase16/endpoints_A.json"      # the output file, relative to the repository root
FIXED = ("p3c_stock", "p3c_c3_original", "ref_stock", "ref_orig")   # stock, the published edit, the two reference arms
# the comparisons given on the command line, (name, stem) in the order given
COMPARE = (("the phase-15 editor's final", "p15_final"), ("our map repair", "p7_uperp_on_loop"))
OTHERS = (("the best blind quick fix (MiMo-flash, phase 4b)", "p4_mimo"), ("the 50-pass edit", "p10_fullbudget"))
CHECKED = (FINAL, REF_FINAL, *FIXED, *(stem for _, stem in COMPARE), *(stem for _, stem in OTHERS))   # as main() checks
NAMES = ("FINAL", "STOCK", "ORIG", "REF_FINAL", "REF_STOCK", "REF_ORIG")    # the runs the phase-14 script names
# m1 to m3 right or not under the runs whose verdicts main() hands on: a different pattern under each
RIGHT = {FINAL: (True, False, True), "p15_final": (False, False, True), "p7_uperp_on_loop": (True, True, False),
         "p4_mimo": (True, True, True), "p10_fullbudget": (False, False, False)}
CLOSED = (True, False, False)                                         # the thinking closed in m1 of the three
# {problem id: ok_rescored} over the rows of each of those re-score files, as main() hands it to the phase-14 script
VERDICTS = {FINAL: {"m1": True, "m2": False, "m3": True},
            "p15_final": {"m1": False, "m2": False, "m3": True},
            "p7_uperp_on_loop": {"m1": True, "m2": True, "m3": False},
            "p4_mimo": {"m1": True, "m2": True, "m3": True},
            "p10_fullbudget": {"m1": False, "m2": False, "m3": False}}
OTHER_STAMP = "phase3c/rescore.py+phase3b/scorer.py#4a1b56a343a4 (ISSUES #19)"   # the frozen checker of the record
CLEAN = dict(soup_pct=0.0, soup_judged=True, admission_tested=True, problems=[])   # a validity record without a problem
FAILED = dict(CLEAN, problems=["soup: 30.0% of generations end in non-Latin script (bar 20.0%)"])
FIGURES = dict(mean_tokens=812.5, cap_hit_rate=25.0)   # the figures a re-score's summary shares with its run file's
# a run file's settings block, for a run of the model under test under the repository's question; and one that does
# not say under which question the run ran
SETTINGS = dict(model="the model under test", reference_question=False)
NO_QUESTION = {k: v for k, v in SETTINGS.items() if k != "reference_question"}


def both_files(rights=(False, False, False)):
    """A run's two files, (the re-score, the run), as base.rescored() and base.run() return them: three problems m1 to
    m3, right per `rights`; a re-score in the tree's checker's stamp, with the thinking read (no_think False), a
    validity record without a problem and the figures of the run file beside it. No settings block, so that the run is
    not checked for kind unless a test gives it one."""
    rows = [dict(id=f"m{i}", ok_rescored=right, closed=closed)
            for i, (right, closed) in enumerate(zip(rights, CLOSED), 1)]
    re_scored = {"summary": dict(name="x_rescored", n=3, rescored_with=base.rescore.RESCORED_WITH, no_think=False,
                                 validity=dict(CLEAN), **FIGURES),
                 "rows": rows}
    run = {"summary": dict(name="x", n=3, wall_s=63.0, **FIGURES)}
    return re_scored, run


def pair(delta, low, high):
    """A pair with what hint_word() reads: the difference in points, and its 95% interval."""
    return dict(delta=delta, ci95=[low, high])


BASE_DOCUMENT = {   # what the phase-14 script builds, in miniature: its keys
    "about": "rescore_file_stamp identifies the checker that produced each re-score file",
    "primary_runs": {"final candidate": {"accuracy": 60.0, "n": 500}, "stock": {"accuracy": 50.0, "n": 500},
                     "published edit": {"accuracy": 45.0, "n": 500}},
    "primary_pairs": [{"label": "final candidate against stock", "delta": 10.0},
                      {"label": "final candidate against the published edit", "delta": 15.0, "word": "EXCEEDS"}],
    "repair": {"holds": True, "right_final": 300, "right_stock": 250, "n": 500},
    "secondary_runs": {"x_ref": {"accuracy": 61.0, "n": 500}},
    "graders": {"checker": "the checker in this tree"},
    "secondary_pairs": [{"label": "final candidate against x, repository matcher", "delta": 8.0}],
}
# the keys the script adds to the phase-14 script's document, besides the extension of "about"
ADDED = ("exploratory_runs", "exploratory_pairs", "other_exploratory_runs", "other_exploratory_pairs")
PAIR = dict(label="the pair() stand-in's result", n=500, first=61.0, second=50.2, delta=10.8, ci95=[6.4, 15.2],
            only_first=75, only_second=22, p=1.2e-08)
ABOUT = (BASE_DOCUMENT["about"] + f" final: {FINAL}; reference run: {REF_FINAL}. exploratory_runs, exploratory_pairs, "
         "other_exploratory_runs and other_exploratory_pairs are comparisons without a protocol word; each pair also "
         "names the run it compares under `stem`.")
HINT_SENTENCE = " `hint` applies the phase-16 rule for the told session to its pairs with the blind finals."


def level(accuracy, right):
    """base.describe()'s level of a run of both_files() with `right` of its three problems right, `accuracy` their
    share in percent; the rest is what every stand-in run carries."""
    return dict(accuracy=accuracy, right=right, n=3, cap_hit_rate=25.0, closed=33.3, mean_tokens=812.5, wall_s=63.0,
                validity=CLEAN, rescore_file_stamp=base.rescore.RESCORED_WITH)


LEVELS = {"p15_final": level(33.3, 1), "p7_uperp_on_loop": level(66.7, 2), "p4_mimo": level(100.0, 3),
          "p10_fullbudget": level(0.0, 0)}   # the compared runs, as base.describe() reads them from both_files()


class NotesTheFile(io.StringIO):
    """What main() prints, and for each of its writes to stdout whether the output file was on disk already."""

    def __init__(self, path):
        super().__init__()
        self.path, self.file_was_there = path, []

    def write(self, text):
        self.file_was_there.append(self.path.exists())
        return super().write(text)


class WithStandIns(unittest.TestCase):
    """A test in which values can be stood in for modules' attributes (stand_in()); tearDown() puts every attribute
    back as it was first found."""

    def setUp(self):
        self.as_they_were = {}     # (module, attribute) → what the attribute held before anything was stood in

    def stand_in(self, module, attribute, value):
        self.as_they_were.setdefault((module, attribute), getattr(module, attribute))
        setattr(module, attribute, value)

    def tearDown(self):
        for (module, attribute), was in self.as_they_were.items():
            setattr(module, attribute, was)


class ParseEntriesTest(unittest.TestCase):
    """parse_entries(): the NAME=STEM entries of --compare and --other, taken together"""

    def parsed(self, *groups):
        """parse_entries() over the options' entries, `groups` in their order."""
        return mod.parse_entries([list(group) for group in groups])

    def exit_text(self, *groups):
        """The text parse_entries() stopped with."""
        with self.assertRaises(SystemExit) as raised:
            self.parsed(*groups)
        return str(raised.exception.code)

    def test_splits_each_entry_at_its_last_equals_sign_and_strips_both_sides(self):
        # a name with spaces around it, then a name that itself holds an equals sign
        self.assertEqual(self.parsed([" the phase-15 editor's final = p15_final ", "a=b=p4_mimo"]),
                         [("p15_final", "the phase-15 editor's final"), ("p4_mimo", "a=b")])

    def test_keeps_the_order_given_across_the_options(self):
        self.assertEqual(self.parsed(["a=p4_mimo"], ["b=p10_fullbudget", "c=p15_final"]),
                         [("p4_mimo", "a"), ("p10_fullbudget", "b"), ("p15_final", "c")])

    def test_an_entry_without_an_equals_sign_stops_with_a_message_naming_it(self):
        self.assertEqual(self.exit_text(["no equals sign"]),
                         "--compare and --other take NAME=STEM; got 'no equals sign'")

    def test_an_entry_with_an_empty_side_stops_with_a_message_naming_it(self):
        for entry in ("=p4_mimo", "name=", " = ", "name= "):
            with self.subTest(entry=entry):
                self.assertEqual(self.exit_text([entry]), f"--compare and --other take NAME=STEM; got {entry!r}")

    def test_a_stem_named_twice_inside_one_option_stops_with_a_message_naming_it_once(self):
        self.assertEqual(self.exit_text(["a=p4_mimo", "b=p4_mimo", "c=p4_mimo"]),
                         "a run is named twice: ['p4_mimo']")

    def test_a_stem_named_in_both_options_stops_with_a_message_naming_it(self):
        self.assertEqual(self.exit_text(["a=p4_mimo"], ["b=p4_mimo"]), "a run is named twice: ['p4_mimo']")

    def test_a_name_named_twice_stops_with_a_message_naming_it(self):
        for groups in ((["same=p4_mimo", "same=p10_fullbudget"],), (["same=p4_mimo"], ["same=p10_fullbudget"])):
            with self.subTest(groups=groups):
                self.assertEqual(self.exit_text(*groups), "a name is named twice: ['same']")


class HintWordTest(unittest.TestCase):
    """hint_word(): the phase-16 rule's word for the told final's pairs with the blind finals"""

    def test_no_pairs_is_inconclusive(self):
        self.assertEqual(mod.hint_word([]), "INCONCLUSIVE")

    def test_the_hint_helps_when_every_pair_is_more_than_four_points_above_with_its_interval_above_zero(self):
        self.assertEqual(mod.hint_word([pair(5.0, 1.0, 9.0), pair(4.1, 0.1, 8.1)]), "THE HINT HELPS")

    def test_the_hint_hurts_when_every_pair_is_more_than_four_points_below_with_its_interval_below_zero(self):
        self.assertEqual(mod.hint_word([pair(-5.0, -9.0, -1.0), pair(-4.1, -8.1, -0.1)]), "THE HINT HURTS")

    def test_no_measurable_effect_when_every_pair_is_within_four_points(self):
        self.assertEqual(mod.hint_word([pair(4.0, -1.0, 9.0), pair(-3.5, -12.0, 5.0), pair(0.0, -2.0, 2.0)]),
                         "NO MEASURABLE EFFECT")

    def test_four_points_above_exactly_is_no_measurable_effect_and_just_above_is_the_hint_helps(self):
        self.assertEqual(mod.hint_word([pair(4.0, 1.0, 7.0)]), "NO MEASURABLE EFFECT")     # not more than 4.0 above
        self.assertEqual(mod.hint_word([pair(4.1, 1.0, 7.1)]), "THE HINT HELPS")

    def test_four_points_below_exactly_is_no_measurable_effect_and_just_below_is_the_hint_hurts(self):
        self.assertEqual(mod.hint_word([pair(-4.0, -7.0, -1.0)]), "NO MEASURABLE EFFECT")
        self.assertEqual(mod.hint_word([pair(-4.1, -7.1, -1.1)]), "THE HINT HURTS")

    def test_an_interval_touching_zero_makes_more_than_four_points_inconclusive(self):
        self.assertEqual(mod.hint_word([pair(6.0, 0.0, 12.0)]), "INCONCLUSIVE")
        self.assertEqual(mod.hint_word([pair(-6.0, -12.0, 0.0)]), "INCONCLUSIVE")

    def test_a_mix_is_inconclusive(self):
        # one pair helps and one is within 4.0 points; one helps and one's interval touches zero
        self.assertEqual(mod.hint_word([pair(5.0, 1.0, 9.0), pair(2.0, -3.0, 7.0)]), "INCONCLUSIVE")
        self.assertEqual(mod.hint_word([pair(5.0, 1.0, 9.0), pair(6.0, 0.0, 12.0)]), "INCONCLUSIVE")


class MainTest(WithStandIns):
    """main(): the arguments, then the kinds of the runs and the checks, then the endpoint script of phase 14 under the
    command-line names, then the comparisons and the hint, then the one write"""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)                 # stands in for the repository root
        self.folder16, self.folder14 = self.root / "phase16", self.root / "phase14"
        for folder in (self.folder16, self.folder14):
            folder.mkdir()
        self.out = self.folder16 / "endpoints_A.json"   # where the default --out puts the file
        self.rescored_files, self.run_files = {}, {}
        for stem in CHECKED:
            self.give_files(stem)
        # no --ref-final, as in most real commands: the reference run is the final's name with _ref by default. Each
        # entry has its own --compare or --other, as the option's help says
        self.argv = ["--final", FINAL, "--out", OUT]
        for option, entries in (("--compare", COMPARE), ("--other", OTHERS)):
            for name, stem in entries:
                self.argv += [option, f"{name}={stem}"]
        # the checks and the build, in order; what the phase-14 script sees when it is built; the calls of pair()
        self.log, self.seen, self.paired = [], [], []

        def reader(files, kind):
            """base.rescored() stood in (kind "rescored"), base.run() (kind "run"): `stem`'s file out of `files`. A
            file that is not there raises the error Path.read_text() raises, with the file's name."""
            def read(stem):
                if stem not in files:
                    name = f"{stem}_rescored.json" if kind == "rescored" else f"{stem}.json"
                    raise FileNotFoundError(2, "No such file or directory", str(Path("phase3c") / name))
                return files[stem]
            return read

        real_checked = mod.checked

        def checked(stem, reference):
            """checked() as it is, each call noted first."""
            self.log.append(("checked", stem))
            return real_checked(stem, reference)

        def build():
            """base.build() stood in: notes the call and what the phase-14 script sees of its runs and folder, and
            returns a small document, a fresh one each time as the real one does."""
            self.log.append(("build", None))
            self.seen.append({name: getattr(base, name) for name in NAMES + ("HERE",)})
            return dict(BASE_DOCUMENT)

        def pair(first, second, label, word_key=None):     # the real pair()'s signature
            self.paired.append((first, second, label, word_key))
            return dict(PAIR)                              # a copy: what main() does to it shows in the file only

        self.built, self.base_main = mock.Mock(side_effect=build), mock.Mock()
        self.described, self.wrote = mock.Mock(wraps=base.describe), mock.Mock(wraps=base.write_atomic)
        self.read_rescored = mock.Mock(name="base.rescored", side_effect=reader(self.rescored_files, "rescored"))
        self.read_run = mock.Mock(name="base.run", side_effect=reader(self.run_files, "run"))
        # hint_word() stood in, where a test is about what main() does with its words; the rule itself is
        # HintWordTest's. Its word says how many pairs it was given
        self.hint_words = mock.Mock(name="hint_word", side_effect=lambda pairs: f"the word over {len(pairs)} pair(s)")
        # stood in: this script's folder, and its check (noted, then run as it is); the phase-14 script's folder and
        # its two final runs (other values, so that the tests can see main() set its own and put them back); its
        # readers, build(), main() and pair(). self.wrote notes the calls of the real write_atomic()
        self.stand_in(mod, "HERE", self.folder16)
        self.stand_in(mod, "checked", checked)
        self.stand_in(base, "HERE", self.folder14)
        self.stand_in(base, "FINAL", "x")
        self.stand_in(base, "REF_FINAL", "x_ref")
        self.at_start = self.names()      # what main() finds, and must leave
        self.stand_in(base, "rescored", self.read_rescored)
        self.stand_in(base, "run", self.read_run)
        self.stand_in(base, "build", self.built)
        self.stand_in(base, "main", self.base_main)
        self.stand_in(base, "describe", self.described)
        self.stand_in(base, "pair", pair)
        self.stand_in(base, "write_atomic", self.wrote)

    def give_files(self, stem):
        """Stand-in files for the run `stem`."""
        self.rescored_files[stem], self.run_files[stem] = both_files(RIGHT.get(stem, (False, False, False)))

    def settings(self, stem, **fields):
        """Give the run file of `stem` a settings block: SETTINGS with `fields` put in."""
        self.run_files[stem]["summary"]["settings"] = dict(SETTINGS, **fields)

    def names(self):
        """The runs and the folder the phase-14 script goes by."""
        return {name: getattr(base, name) for name in NAMES + ("HERE",)}

    def run_main(self, argv=None):
        """main(argv), by default with self.argv; what it prints is kept out of the test run's output, in
        self.printed, what it puts on stderr in self.errs, and in self.file_before_print, for each write to stdout,
        whether the output file was on disk."""
        printed, errs = NotesTheFile(self.out), io.StringIO()
        try:
            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(errs):
                mod.main(self.argv if argv is None else argv)
        finally:
            self.printed, self.errs = printed.getvalue(), errs.getvalue()
            self.file_before_print = printed.file_was_there

    def exit_code(self, argv=None):
        """What main(argv) stopped with: the text of sys.exit(<text>), or argparse's status for a usage error."""
        with self.assertRaises(SystemExit) as raised:
            self.run_main(argv)
        return raised.exception.code

    def exit_text(self, argv=None):
        """The text main(argv) stopped with (sys.exit(<text>): on the real command line it goes to stderr, status 1)."""
        return str(self.exit_code(argv))

    def written(self):
        """The output file of this run, read back."""
        return json.loads(self.out.read_text(encoding="utf-8"))

    def assert_nothing_written(self):
        """The phase-14 script's main() and write_atomic() did not run, nothing was printed, and the tree holds its two
        folders alone: no output file, and no .tmp beside it."""
        self.base_main.assert_not_called()
        self.wrote.assert_not_called()
        self.assertEqual(self.printed, "")
        self.assertEqual(sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")),
                         ["phase14", "phase16"])

    def assert_stopped_before_the_build(self):
        self.built.assert_not_called()
        self.assert_nothing_written()

    def assert_nothing_read_or_built(self):
        self.read_rescored.assert_not_called()
        self.read_run.assert_not_called()
        self.assert_stopped_before_the_build()

    def assert_stopped_by_the_kinds(self):
        """main() stopped in check_kinds(): no run has been checked, nothing was built, written or printed, and the
        names are back as they were."""
        self.assertEqual(self.log, [])
        self.assert_stopped_before_the_build()
        self.assertEqual(self.names(), self.at_start)

    # ---- the arguments ----
    def test_stops_with_a_usage_error_when_final_or_out_is_not_given(self):
        for given, missing in (([], "--final"), (["--final", FINAL], "--out")):
            with self.subTest(given=given):
                self.assertEqual(self.exit_code(given), 2)     # argparse's status for a usage error
                self.assertIn(f"the following arguments are required: {missing}", self.errs)
                self.assert_nothing_read_or_built()

    def test_an_empty_or_blank_final_stops_with_a_message_before_anything_is_read(self):
        for given in ("", "   "):
            with self.subTest(final=given):
                self.assertEqual(self.exit_text(["--final", given, "--out", OUT]), "--final must name a run")
                self.assert_nothing_read_or_built()
                self.assertEqual(self.names(), self.at_start)

    def test_the_whitespace_around_final_ref_final_and_out_is_stripped(self):
        self.run_main(["--final", f" {FINAL} ", "--ref-final", f" {REF_FINAL} ", "--out", f" {OUT} "])
        self.assertEqual(self.seen, [dict(self.at_start, FINAL=FINAL, REF_FINAL=REF_FINAL, HERE=self.folder16)])
        self.wrote.assert_called_once_with(self.out, mock.ANY)
        self.assertIn(f" final: {FINAL}; reference run: {REF_FINAL}.", self.written()["about"])

    def test_ref_final_names_the_reference_run(self):
        # by default, and with --ref-final "", the reference run is the final's name with _ref (the two tests below)
        self.give_files("p16_A_final_other_ref")
        self.run_main(self.argv + ["--ref-final", "p16_A_final_other_ref"])
        self.assertEqual(self.seen, [dict(self.at_start, FINAL=FINAL, REF_FINAL="p16_A_final_other_ref",
                                          HERE=self.folder16)])
        self.assertIn(("checked", "p16_A_final_other_ref"), self.log)
        self.assertIn("reference run: p16_A_final_other_ref", self.written()["about"])

    def test_an_empty_ref_final_means_the_final_s_name_with_ref(self):
        self.run_main(["--final", FINAL, "--ref-final", "", "--out", OUT])
        self.assertEqual(self.seen, [dict(self.at_start, FINAL=FINAL, REF_FINAL=REF_FINAL, HERE=self.folder16)])
        self.assertIn("reference run: p16_A_final_ref", self.written()["about"])

    def test_without_ref_final_the_reference_run_is_the_final_s_name_with_ref(self):
        self.run_main(["--final", FINAL, "--out", OUT])
        self.assertEqual(self.seen, [dict(self.at_start, FINAL=FINAL, REF_FINAL=REF_FINAL, HERE=self.folder16)])
        self.assertIn("reference run: p16_A_final_ref", self.written()["about"])

    def test_a_malformed_or_repeated_entry_stops_main_before_anything_is_read_or_built(self):
        cases = ((["--compare", "no equals sign"], "--compare and --other take NAME=STEM; got 'no equals sign'"),
                 (["--other", "=p4_mimo"], "--compare and --other take NAME=STEM; got '=p4_mimo'"),
                 (["--compare", "a=p4_mimo", "--compare", "b=p4_mimo", "--compare", "c=p4_mimo"],
                  "a run is named twice: ['p4_mimo']"))     # each listed once
        for extra, expected in cases:
            with self.subTest(extra=extra):
                self.assertEqual(self.exit_text(self.argv + extra), expected)
                self.assert_nothing_read_or_built()
                self.assertEqual(self.names(), self.at_start)     # it stops before main() touches the names

    def test_a_stem_named_in_both_compare_and_other_stops_with_a_message_naming_it(self):
        self.assertEqual(self.exit_text(["--final", FINAL, "--out", OUT,
                                         "--compare", "a=p4_mimo", "--other", "b=p4_mimo"]),
                         "a run is named twice: ['p4_mimo']")
        self.assert_nothing_read_or_built()

    def test_a_name_named_in_both_compare_and_other_stops_with_a_message_naming_it(self):
        self.assertEqual(self.exit_text(["--final", FINAL, "--out", OUT,
                                         "--compare", "same=p4_mimo", "--other", "same=p10_fullbudget"]),
                         "a name is named twice: ['same']")
        self.assert_nothing_read_or_built()

    def test_hint_versus_naming_a_run_that_is_not_a_compare_entry_stops_before_anything_is_read(self):
        # p4_mimo is an --other entry; nowhere is not given at all
        self.assertEqual(self.exit_text(self.argv + ["--hint-versus", "p4_mimo", "--hint-versus", "nowhere"]),
                         "--hint-versus names runs that are not --compare entries: ['p4_mimo', 'nowhere']")
        self.assert_nothing_read_or_built()
        self.assertEqual(self.names(), self.at_start)

    # ---- the rule for --out ----
    def test_a_relative_out_is_taken_from_the_repository_root(self):
        self.run_main()
        self.wrote.assert_called_once_with(self.root / OUT, mock.ANY)

    def test_an_absolute_out_in_the_script_s_own_folder_is_taken_as_given(self):
        self.run_main(["--final", FINAL, "--out", str(self.out)])
        self.wrote.assert_called_once_with(self.out, mock.ANY)

    def test_an_out_that_is_not_a_file_in_the_script_s_own_folder_stops_before_anything_is_read(self):
        elsewhere = self.root.parent / "elsewhere" / "x.json"     # absolute, and outside the repository
        # another phase's report, a folder below the script's, the repository root itself, a folder, a folder elsewhere
        for given in ("phase15/endpoints.json", "phase16/sub/x.json", "x.json", "phase16", str(elsewhere)):
            with self.subTest(out=given):
                rule = ("--out must be a file in this script's own folder, relative to the repository root "
                        "(such as phase16/endpoints_A.json), so that it can never overwrite another phase's report")
                self.assertEqual(self.exit_text(["--final", FINAL, "--out", given]), f"{rule}; got {given}")
                self.assert_nothing_read_or_built()
                self.assertEqual(self.names(), self.at_start)

    def test_a_path_that_climbs_out_of_the_script_s_folder_and_back_into_it_is_accepted(self):
        # the folder asked for is compared once both paths are resolved
        given = "phase16/../phase16/endpoints_A.json"
        self.run_main(["--final", FINAL, "--out", given])
        self.wrote.assert_called_once_with(self.root / given, mock.ANY)     # written through the path as given, ...
        self.assertEqual([p.name for p in self.folder16.iterdir()], ["endpoints_A.json"])     # ... which lands there

    # ---- the names main() runs under ----
    def test_while_it_runs_the_phase_14_script_s_names_are_the_command_line_names_and_the_output_s_folder(self):
        # the final candidate and its reference take the command-line names (the reference: the final's with _ref when
        # --ref-final is not given); stock, the published edit and the two other reference arms keep theirs; the
        # folder is the output's. The phase-14 script is built once, under them
        self.run_main()
        self.built.assert_called_once_with()
        self.assertEqual(self.seen, [dict(self.at_start, FINAL=FINAL, REF_FINAL=REF_FINAL, HERE=self.folder16)])

    def test_puts_the_names_back_after_it_returns(self):
        self.run_main()
        self.assertEqual(self.names(), self.at_start)

    def test_puts_the_names_back_after_it_stops(self):
        self.rescored_files["ref_stock"]["summary"]["rescored_with"] = OTHER_STAMP
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual(self.names(), self.at_start)

    # ---- the kinds of the runs: checked once the final's files are read, before any run is checked ----
    def test_the_kinds_are_checked_after_the_final_s_files_are_read_and_before_the_per_run_checks(self):
        self.settings(FINAL, reference_question=True)          # a wrong kind, to stop inside check_kinds()
        self.exit_text()
        # the final's files are read first, for its way of reading the thinking; inside check_kinds() only the
        # reference run's files are read before the stop; and no run has been checked yet
        self.assertEqual(self.read_rescored.call_args_list[0], mock.call(FINAL))
        self.assertEqual(self.read_run.call_args_list[0], mock.call(FINAL))
        self.assertEqual({call.args[0] for call in self.read_run.call_args_list}, {FINAL, REF_FINAL})
        self.assertEqual(self.log, [])
        self.assert_stopped_before_the_build()
        self.assertEqual(self.names(), self.at_start)

    def test_a_ref_final_that_is_the_final_s_own_run_stops_with_a_message(self):
        self.assertEqual(self.exit_text(["--final", FINAL, "--ref-final", f" {FINAL} ", "--out", OUT]),
                         f"--ref-final names the final's own run ({FINAL}): the reference run is another run")
        self.assert_stopped_by_the_kinds()

    def test_a_compared_stem_that_is_the_final_its_reference_run_or_a_fixed_run_stops_with_a_message(self):
        for stem in (FINAL, REF_FINAL, *FIXED):
            with self.subTest(stem=stem):
                self.assertEqual(self.exit_text(self.argv + ["--compare", f"again={stem}"]),
                                 f"{stem} is compared with itself: it is the final, its reference run, or one of the "
                                 "fixed runs")
                self.assert_stopped_by_the_kinds()

    def test_a_reference_run_of_another_model_stops_with_a_message_naming_both_models(self):
        self.settings(FINAL, model="model A")
        self.settings(REF_FINAL, model="model B", reference_question=True)
        self.assertEqual(self.exit_text(), f"{FINAL} and {REF_FINAL} were not runs of the same model "
                                           "(model A and model B)")
        self.assert_stopped_by_the_kinds()

    def test_a_final_and_a_reference_run_of_the_same_model_under_their_questions_are_accepted(self):
        # so is every compared run under the repository's question
        self.settings(FINAL)
        self.settings(REF_FINAL, reference_question=True)
        for _, stem in COMPARE + OTHERS:
            self.settings(stem)
        self.run_main()
        self.built.assert_called_once_with()

    def test_a_final_whose_run_file_does_not_say_it_ran_under_the_repository_s_question_stops(self):
        for question in (True, None):      # the reference question; a block that does not say
            with self.subTest(reference_question=question):
                self.run_files[FINAL]["summary"]["settings"] = dict(NO_QUESTION, reference_question=question)
                self.assertEqual(self.exit_text(), f"{FINAL} was not run under the repository's question "
                                                   f"(reference_question={question})")
                self.assert_stopped_by_the_kinds()

    def test_a_reference_run_that_did_not_run_under_the_reference_question_stops(self):
        for question in (False, None):     # the repository's question; a block that does not say
            with self.subTest(reference_question=question):
                self.run_files[REF_FINAL]["summary"]["settings"] = dict(NO_QUESTION, reference_question=question)
                self.assertEqual(self.exit_text(), f"{REF_FINAL} was not run under the reference question "
                                                   f"(reference_question={question})")
                self.assert_stopped_by_the_kinds()

    def test_a_compared_run_that_ran_under_the_reference_question_stops_with_a_message_naming_it(self):
        for _, stem in COMPARE + OTHERS:
            with self.subTest(run=stem):
                # one run at a time: a run left wrong would be named first in the subtests that follow
                summary = self.run_files[stem]["summary"]
                before = summary.get("settings")
                self.settings(stem, reference_question=True)
                self.assertEqual(self.exit_text(), f"{stem} was run under the reference question, so it is not "
                                 "comparable with the final under the repository's question")
                self.assert_stopped_by_the_kinds()
                if before is None:
                    summary.pop("settings", None)
                else:
                    summary["settings"] = before

    def test_a_run_file_without_a_settings_block_is_not_checked_for_kind(self):
        # the final records no settings, so its model is not compared with the reference run's, nor is its question
        # read; a compared run that records none is not asked for its question either
        self.settings(REF_FINAL, model="another model", reference_question=True)
        self.run_main()
        self.built.assert_called_once_with()

    # ---- refused before anything is built: the runs are not one reading ----
    def test_another_checkers_stamp_on_any_of_the_runs_stops_main_naming_the_run_and_both_stamps(self):
        self.assertNotEqual(OTHER_STAMP, base.rescore.RESCORED_WITH)   # the frozen checker of the record
        for stem in CHECKED:
            summary = self.rescored_files[stem]["summary"]
            summary["rescored_with"] = OTHER_STAMP
            with self.subTest(run=stem):
                text = self.exit_text()
                self.assertIn(stem, text)                                # the run
                self.assertIn(OTHER_STAMP, text)                         # the checker it was re-scored with
                self.assertIn(base.rescore.RESCORED_WITH, text)          # the checker in this tree
                self.assert_stopped_before_the_build()
            summary["rescored_with"] = base.rescore.RESCORED_WITH

    def test_a_re_score_that_records_no_checker_stops_main_naming_the_run_and_the_checker_in_this_tree(self):
        for stem in CHECKED:
            summary = self.rescored_files[stem]["summary"]
            stamp = summary.pop("rescored_with")
            with self.subTest(run=stem):
                text = self.exit_text()
                self.assertIn(f"{stem} was re-scored with None", text)
                self.assertIn(base.rescore.RESCORED_WITH, text)
                self.assert_stopped_before_the_build()
            summary["rescored_with"] = stamp

    def test_a_re_score_with_another_no_think_than_the_final_candidate_s_stops_main_naming_both_readings(self):
        # every run but the final candidate itself, whose reading is the one compared with
        for stem in [s for s in CHECKED if s != FINAL]:
            summary = self.rescored_files[stem]["summary"]
            summary["no_think"] = True
            with self.subTest(run=stem):
                text = self.exit_text()
                self.assertIn(f"{stem} was re-scored with no_think=True", text)          # the run and its reading
                self.assertIn("the final candidate with no_think=False: not one reading", text)
                self.assert_stopped_before_the_build()
            summary["no_think"] = False

    def test_a_final_candidate_that_alone_reads_another_no_think_stops_main_at_the_first_run_compared_with_it(self):
        # the reading compared with is the final candidate's: the next run checked, its reference
        self.rescored_files[FINAL]["summary"]["no_think"] = True
        self.assertIn(f"{REF_FINAL} was re-scored with no_think=False, the final candidate with no_think=True",
                      self.exit_text())
        self.assert_stopped_before_the_build()

    def test_a_re_score_whose_figures_are_not_its_run_file_s_stops_main_naming_the_run_the_figure_and_both_values(self):
        for stem in CHECKED:
            summary = self.run_files[stem]["summary"]
            for figure, other in (("mean_tokens", 900.0), ("cap_hit_rate", 30.0)):
                summary[figure] = other
                with self.subTest(run=stem, figure=figure):
                    text = self.exit_text()
                    self.assertIn(f"{stem}: its re-score reads {figure} {FIGURES[figure]} and its run file {other}",
                                  text)
                    self.assert_stopped_before_the_build()
                summary[figure] = FIGURES[figure]

    def test_a_run_with_a_file_that_is_not_there_stops_main_naming_the_run_and_the_file(self):
        # wherever the file is first read: the final's before the checks, the reference run's and the compared runs'
        # in check_kinds(), the rest in a check. All through files(), so all with the same message
        missing = (("rescored", "rescored_files", "(re-score the run first)"),
                   ("run", "run_files", "(the run's files are not collected)"))
        for stem in CHECKED:
            for kind, where, hint in missing:
                files = getattr(self, where)
                name = f"{stem}_rescored.json" if kind == "rescored" else f"{stem}.json"
                file = files.pop(stem)
                with self.subTest(run=stem, file=name):
                    # the reader's error becomes the message; there is no traceback; the hint names what is missing
                    self.assertEqual(self.exit_text(), f"{stem}: No such file or directory: {name} {hint}")
                    self.assert_stopped_before_the_build()
                files[stem] = file

    def test_a_final_candidate_without_a_re_score_file_stops_main_with_the_message_naming_it(self):
        # the final candidate's files are read before the checks, for its way of reading the thinking: through the
        # same files(), so a missing one stops the same way as inside a check
        del self.rescored_files[FINAL]
        self.assertEqual(self.exit_text(),
                         f"{FINAL}: No such file or directory: {FINAL}_rescored.json (re-score the run first)")
        self.assert_stopped_before_the_build()

    # ---- the checks, then the build ----
    def test_checks_the_final_candidate_first_then_each_other_run_once_and_builds_only_then(self):
        # the final candidate, its reference run, stock, the published edit and the two reference arms, then the
        # compared runs and the other runs in the order given; then the build
        self.run_main()
        self.assertEqual(self.log, [("checked", stem) for stem in (FINAL, REF_FINAL, "p3c_stock", "p3c_c3_original",
                                                                    "ref_stock", "ref_orig", "p15_final",
                                                                    "p7_uperp_on_loop", "p4_mimo", "p10_fullbudget")]
                         + [("build", None)])

    # ---- the comparisons ----
    def test_exploratory_runs_holds_each_compared_run_under_its_name_as_the_phase_14_script_describes_it(self):
        self.run_main()
        self.assertEqual(self.described.call_args_list[:len(COMPARE)],
                         [mock.call(stem, VERDICTS[stem]) for _, stem in COMPARE])
        self.assertEqual(self.written()["exploratory_runs"], {name: LEVELS[stem] for name, stem in COMPARE})
        self.assertEqual(list(self.written()["exploratory_runs"]), [name for name, _ in COMPARE])

    def test_other_exploratory_runs_hold_each_other_run_under_its_name_as_the_phase_14_script_describes_it(self):
        # after the compared runs: the other runs, {name: level}, the names in the order given
        self.run_main()
        self.assertEqual(self.described.call_args_list[len(COMPARE):],
                         [mock.call(stem, VERDICTS[stem]) for _, stem in OTHERS])
        self.assertEqual(self.written()["other_exploratory_runs"], {name: LEVELS[stem] for name, stem in OTHERS})
        self.assertEqual(list(self.written()["other_exploratory_runs"]), [name for name, _ in OTHERS])

    def test_exploratory_pairs_pair_the_final_s_verdicts_with_each_compared_run_s_under_its_name(self):
        self.run_main()
        self.assertEqual(self.paired[:len(COMPARE)],
                         [(VERDICTS[FINAL], VERDICTS[stem], f"final candidate against {name}", None)
                          for name, stem in COMPARE])

    def test_other_exploratory_pairs_pair_the_final_s_verdicts_with_each_other_run_s_under_its_name(self):
        self.run_main()
        self.assertEqual(self.paired[len(COMPARE):],
                         [(VERDICTS[FINAL], VERDICTS[stem], f"final candidate against {name}", None)
                          for name, stem in OTHERS])

    def test_each_pair_is_written_as_pair_returned_it_with_its_run_s_stem_added(self):
        self.run_main()
        written = self.written()
        self.assertEqual(written["exploratory_pairs"], [dict(PAIR, stem=stem) for _, stem in COMPARE])
        self.assertEqual(written["other_exploratory_pairs"], [dict(PAIR, stem=stem) for _, stem in OTHERS])

    def test_no_pair_is_asked_for_a_protocol_word(self):
        # pair() takes the protocol's word as a fourth argument (word_key); none is given
        self.run_main()
        self.assertEqual([word_key for *_, word_key in self.paired], [None] * (len(COMPARE) + len(OTHERS)))

    def test_without_compare_and_other_their_four_keys_are_empty_and_nothing_is_described_or_paired(self):
        self.run_main(["--final", FINAL, "--out", OUT])
        written = self.written()
        self.assertEqual((written["exploratory_runs"], written["exploratory_pairs"],
                          written["other_exploratory_runs"], written["other_exploratory_pairs"]), ({}, [], {}, []))
        self.described.assert_not_called()
        self.assertEqual(self.paired, [])

    def test_the_output_holds_the_phase_14_script_s_document_with_the_four_keys_added(self):
        self.run_main()
        written = self.written()
        self.assertEqual(sorted(written), sorted([*BASE_DOCUMENT, *ADDED]))
        self.assertEqual({k: v for k, v in written.items() if k not in ADDED and k != "about"},
                         {k: v for k, v in BASE_DOCUMENT.items() if k not in ADDED and k != "about"})

    def test_the_about_text_is_the_base_text_extended_naming_the_final_and_the_reference_run(self):
        self.run_main()
        self.assertEqual(self.written()["about"], ABOUT)

    # ---- the hint, for the told session ----
    def test_hint_holds_the_rule_the_word_against_each_named_run_and_the_word_over_all_of_them(self):
        # the words come from hint_word(): once with each named run's pair alone, in the order given, then once with
        # all of them. The named runs are stripped like the other names
        self.stand_in(mod, "hint_word", self.hint_words)
        self.run_main(self.argv + ["--hint-versus", "p7_uperp_on_loop", "--hint-versus", " p15_final "])
        hint = self.written()["hint"]
        self.assertEqual(hint, dict(rule=mod.HINT_RULE,
                                    against={"p7_uperp_on_loop": "the word over 1 pair(s)",
                                             "p15_final": "the word over 1 pair(s)"},
                                    word="the word over 2 pair(s)"))
        self.assertEqual(list(hint["against"]), ["p7_uperp_on_loop", "p15_final"])
        self.assertEqual(self.hint_words.call_args_list,
                         [mock.call([dict(PAIR, stem="p7_uperp_on_loop")]), mock.call([dict(PAIR, stem="p15_final")]),
                          mock.call([dict(PAIR, stem="p7_uperp_on_loop"), dict(PAIR, stem="p15_final")])])
        for word in ("THE HINT HELPS", "THE HINT HURTS", "NO MEASURABLE EFFECT", "INCONCLUSIVE"):
            self.assertIn(word, hint["rule"])                  # the rule's text names its four words

    def test_without_hint_versus_there_is_no_hint_key(self):
        self.run_main()
        self.assertNotIn("hint", self.written())

    def test_the_about_text_ends_with_the_sentence_that_names_the_hint(self):
        self.stand_in(mod, "hint_word", self.hint_words)
        self.run_main(self.argv + ["--hint-versus", "p15_final"])
        self.assertEqual(self.written()["about"], ABOUT + HINT_SENTENCE)

    def test_through_main_the_hint_s_words_are_the_rule_s_words_for_the_told_final_s_pairs(self):
        # every stand-in pair lies 10.8 points above, with its interval from 6.4 to 15.2: the rule's best word
        self.run_main(self.argv + ["--hint-versus", "p15_final", "--hint-versus", "p7_uperp_on_loop"])
        self.assertEqual(self.written()["hint"], dict(rule=mod.HINT_RULE,
                                                      against={"p15_final": "THE HINT HELPS",
                                                               "p7_uperp_on_loop": "THE HINT HELPS"},
                                                      word="THE HINT HELPS"))

    # ---- the one write, the one print ----
    def test_writes_the_file_once_itself_before_anything_is_printed(self):
        self.run_main()
        self.wrote.assert_called_once_with(self.out, mock.ANY)
        self.base_main.assert_not_called()                 # the phase-14 script's own main() is never run
        self.assertEqual([p.name for p in self.folder16.iterdir()], ["endpoints_A.json"])   # no .tmp left beside it
        self.assertTrue(self.file_before_print)            # something was printed ...
        self.assertTrue(all(self.file_before_print), self.file_before_print)   # ... and only with the file on disk

    def test_the_printed_text_is_the_file_s_text(self):
        self.run_main()
        self.assertEqual(self.printed, self.out.read_text(encoding="utf-8"))

    # ---- a failure after the build leaves no output file ----
    def test_a_refusal_inside_a_pair_leaves_no_output_file(self):
        def refusing(first, second, label, word_key=None):
            sys.exit(f"{label}: the two runs do not hold the same 500 problems")   # what the real pair() does then

        self.stand_in(base, "pair", refusing)
        with self.assertRaises(SystemExit):
            self.run_main()
        self.built.assert_called_once_with()             # the refusal came after the build
        self.assert_nothing_written()

    def test_a_compared_run_that_is_not_a_measurement_leaves_no_output_file(self):
        # the checks read no validity record; base.describe(), which the runs come from, refuses the run
        self.rescored_files["p15_final"]["summary"]["validity"] = FAILED
        text = self.exit_text()
        self.assertEqual(text, f"p15_final is not a measurement: its validity record reads {FAILED} (ISSUES #20)")
        self.built.assert_called_once_with()
        self.assert_nothing_written()

    def test_an_error_after_the_build_leaves_no_output_file(self):
        self.described.side_effect = OSError("the re-score file of a compared run cannot be read")
        with self.assertRaises(OSError):
            self.run_main()
        self.built.assert_called_once_with()
        self.assert_nothing_written()


class ImportTest(WithStandIns):
    """Importing phase16/compute_endpoints.py"""

    def setUp(self):
        super().setUp()
        # whatever the import would call on the phase-14 script is noted instead of run
        self.calls = {name: mock.Mock(name=f"base.{name}")
                      for name in ("main", "build", "rescored", "run", "describe", "pair", "write_atomic")}
        for name, call in self.calls.items():
            self.stand_in(base, name, call)
        self.path = list(sys.path)     # the import puts phase14/ on sys.path; what it adds is taken off again

    def tearDown(self):
        super().tearDown()
        sys.path[:] = self.path

    def test_importing_the_script_leaves_the_phase_14_script_s_names_and_folder_as_they_are(self):
        # phase14/endpoints.py's own: its final candidate p14_final, its reference p14_final_ref, its folder
        load_script()
        self.assertEqual({name: getattr(base, name) for name in ("FINAL", "REF_FINAL", "HERE")},
                         dict(FINAL="p14_final", REF_FINAL="p14_final_ref", HERE=HERE.parent / "phase14"))

    def test_importing_the_script_runs_nothing(self):
        load_script()
        for name, call in self.calls.items():
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
