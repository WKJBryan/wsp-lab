"""Tests for phase15/compute_endpoints.py, which runs the endpoint script of phase 14 (phase14/endpoints.py, tested in
phase14/test_endpoints.py) under the names of this phase's runs and writes the endpoints file into this folder once,
with this phase's exploratory additions: the level of the phase-14 editor's final and the pair against it, and the
levels of four other versions of the model and the pairs against them.
  ImportTest  importing the script: it runs nothing, and leaves the phase-14 script as that script made itself;
  MainTest    the checks before anything is built (the tree's checker, one way of reading the thinking, figures that
              agree with the run file, files that are there), on the seven runs and then on the four other versions;
              then the build under this phase's names and folder; the additions; the one write and the one print;
              the names put back; a failure after the build, which leaves no endpoints file.

No run file is read, and nothing is written into phase14/ or phase15/. Stood in: this script's folder; the phase-14
script's folder; the phase-14 script's readers rescored() and run(), which answer from small files; its build(),
which returns a small document; its main(); and its pair(), which returns a fixed result. Real: the script's
checked() (each call noted first), and the phase-14 script's describe() and write_atomic() (their calls noted).
main() sets FINAL, REF_FINAL and HERE on the phase-14 script and puts them back itself; the tests stand other values
in first, and tearDown() puts every stood-in attribute back as it was: the phase-14 script is one module object,
shared with its own tests when both suites run in one process.
Standard library only.
Usage (from the repository root):
  python -B -m unittest -v phase15/test_compute_endpoints.py
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
    """phase15/compute_endpoints.py as a module object, imported from its file under a name of its own and so
    executed afresh. Not under "endpoints", the name it gives the phase-14 script: the phase-14 tests import that
    script under it, and both suites must get the one module object."""
    spec = importlib.util.spec_from_file_location("compute_endpoints", HERE / "compute_endpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_script()
base = mod.base   # phase14/endpoints.py: the script that compute_endpoints.py wraps

FINAL, REF_FINAL, PREVIOUS = "p15_final", "p15_final_ref", "p14_final"
# other versions of the model that the record compares the final candidate with, as compute_endpoints.py names them
OTHERS = (("p4_mimo", "the best blind quick fix (MiMo-flash, phase 4b)"),
          ("p10_fullbudget", "the 50-pass edit (phase 10)"),
          ("p7_stack_full_v2", "the published edit with our repair at full strength (phase 7)"),
          ("p7_uperp_on_loop", "our map repair (phase 7)"))
# every run the script names: this phase's final candidate and its reference, stock, the published edit and the two
# other reference arms (the phase-14 script's order), the two finals of phase 14, and the four other versions
ALL_RUNS = (FINAL, "p3c_stock", "p3c_c3_original", REF_FINAL, "ref_stock", "ref_orig", PREVIOUS, "p14_final_ref",
            *(stem for stem, _ in OTHERS))
NAMES = ("FINAL", "STOCK", "ORIG", "REF_FINAL", "REF_STOCK", "REF_ORIG")    # the runs the phase-14 script names
# m1 to m3 right or not under the runs main() describes and pairs: a different pattern under each
RIGHT = {FINAL: (True, False, True), PREVIOUS: (False, False, True), "p4_mimo": (False, True, True),
         "p10_fullbudget": (False, False, False), "p7_stack_full_v2": (True, True, True),
         "p7_uperp_on_loop": (True, False, False)}
CLOSED = (True, False, False)                                         # the thinking closed in m1 of the three
# {problem id: ok_rescored} over the rows of each of those re-score files, as main() hands it to the phase-14 script
VERDICTS = {FINAL: {"m1": True, "m2": False, "m3": True}, PREVIOUS: {"m1": False, "m2": False, "m3": True},
            "p4_mimo": {"m1": False, "m2": True, "m3": True},
            "p10_fullbudget": {"m1": False, "m2": False, "m3": False},
            "p7_stack_full_v2": {"m1": True, "m2": True, "m3": True},
            "p7_uperp_on_loop": {"m1": True, "m2": False, "m3": False}}
OTHER_STAMP = "phase3c/rescore.py+phase3b/scorer.py#4a1b56a343a4 (ISSUES #19)"   # the frozen checker of the record
CLEAN = dict(soup_pct=0.0, soup_judged=True, admission_tested=True, problems=[])   # a validity record without a problem
FAILED = dict(CLEAN, problems=["soup: 30.0% of generations end in non-Latin script (bar 20.0%)"])
FIGURES = dict(mean_tokens=812.5, cap_hit_rate=25.0)   # the figures a re-score's summary shares with its run file's
# how a run of the stand-ins fails a check: the summary the field is in, the field, the value that fails and the one
# that passes, and the words of the message that say which check stopped main()
FAILINGS = (("re_score", "rescored_with", OTHER_STAMP, base.rescore.RESCORED_WITH, "re-score it first"),
            ("re_score", "no_think", True, False, "not one reading"),
            ("run", "mean_tokens", 900.0, FIGURES["mean_tokens"], "generated again"))


def both_files(rights=(False, False, False)):
    """A run's two files, (the re-score, the run), as base.rescored() and base.run() return them: three problems m1 to
    m3, right per `rights`; a re-score in the tree's checker's stamp, with the thinking read (no_think False), a
    validity record without a problem and the figures of the run file beside it."""
    rows = [dict(id=f"m{i}", ok_rescored=right, closed=closed)
            for i, (right, closed) in enumerate(zip(rights, CLOSED), 1)]
    re_scored = {"summary": dict(name="x_rescored", n=3, rescored_with=base.rescore.RESCORED_WITH, no_think=False,
                                 validity=dict(CLEAN), **FIGURES),
                 "rows": rows}
    run = {"summary": dict(name="x", n=3, wall_s=63.0, **FIGURES)}
    return re_scored, run


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
# the keys this phase adds to the phase-14 script's document, besides the extension of "about"
ADDED = ("exploratory_runs", "exploratory_pairs", "other_exploratory_runs", "other_exploratory_pairs")
PAIR = dict(label="the pair() stand-in's result", n=500, first=61.0, second=50.2, delta=10.8, ci95=[6.4, 15.2],
            only_first=75, only_second=22, p=1.2e-08)


def level(accuracy, right):
    """base.describe()'s level of a run of both_files() with `right` of its three problems right, `accuracy` their
    share in percent; the rest is what every stand-in run carries."""
    return dict(accuracy=accuracy, right=right, n=3, cap_hit_rate=25.0, closed=33.3, mean_tokens=812.5, wall_s=63.0,
                validity=CLEAN, rescore_file_stamp=base.rescore.RESCORED_WITH)


LEVELS = {PREVIOUS: level(33.3, 1), "p4_mimo": level(66.7, 2), "p10_fullbudget": level(0.0, 0),
          "p7_stack_full_v2": level(100.0, 3), "p7_uperp_on_loop": level(33.3, 1)}   # the runs main() describes


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


class MainTest(WithStandIns):
    """main(): the checks, then the endpoint script of phase 14 under this phase's names, then this phase's additions,
    then the one write"""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder15, self.folder14 = Path(tmp.name) / "phase15", Path(tmp.name) / "phase14"
        for folder in (self.folder15, self.folder14):
            folder.mkdir()
        self.rescored_files, self.run_files = {}, {}
        for stem in ALL_RUNS:
            self.rescored_files[stem], self.run_files[stem] = both_files(RIGHT.get(stem, (False, False, False)))
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

        def checked(stem):
            """checked() as it is, each call noted first."""
            self.log.append(("checked", stem))
            return real_checked(stem)

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
        # stood in: this script's folder, and its check (noted, then run as it is); the phase-14 script's folder and
        # its two final runs (other values, so that the tests can see main() set its own and put them back); its
        # readers, build(), main() and pair(). self.wrote notes the calls of the real write_atomic()
        self.stand_in(mod, "HERE", self.folder15)
        self.stand_in(mod, "checked", checked)
        self.stand_in(base, "HERE", self.folder14)
        self.stand_in(base, "FINAL", "x")
        self.stand_in(base, "REF_FINAL", "x_ref")
        self.at_start = {name: getattr(base, name) for name in NAMES + ("HERE",)}   # what main() finds, and must leave
        self.stand_in(base, "rescored", reader(self.rescored_files, "rescored"))
        self.stand_in(base, "run", reader(self.run_files, "run"))
        self.stand_in(base, "build", self.built)
        self.stand_in(base, "main", self.base_main)
        self.stand_in(base, "describe", self.described)
        self.stand_in(base, "pair", pair)
        self.stand_in(base, "write_atomic", self.wrote)

    def run_main(self):
        """main(), with what it prints kept out of the test run's output and kept in self.printed."""
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                mod.main()
        finally:
            self.printed = printed.getvalue()

    def exit_text(self):
        """The text main() stopped with (sys.exit(<text>): on the real command line it goes to stderr, status 1)."""
        with self.assertRaises(SystemExit) as raised:
            self.run_main()
        return str(raised.exception.code)

    def written(self):
        """The endpoints file of this phase's folder, read back."""
        return json.loads((self.folder15 / "endpoints.json").read_text(encoding="utf-8"))

    def assert_nothing_written(self):
        """The phase-14 script's main() and write_atomic() did not run, nothing was printed, and neither folder holds
        a file: no endpoints.json, and no .tmp beside it."""
        self.base_main.assert_not_called()
        self.wrote.assert_not_called()
        self.assertEqual(self.printed, "")
        self.assertEqual([p.name for p in self.folder15.iterdir()], [])
        self.assertEqual([p.name for p in self.folder14.iterdir()], [])

    def assert_stopped_before_the_build(self):
        self.built.assert_not_called()
        self.assert_nothing_written()

    # ---- refused before anything is built: the runs are not one reading ----
    def test_another_checkers_stamp_on_any_of_the_runs_stops_main_naming_the_run_and_both_stamps(self):
        self.assertNotEqual(OTHER_STAMP, base.rescore.RESCORED_WITH)   # the frozen checker of the record
        for stem in dict.fromkeys(mod.RUNS):                           # each run of RUNS once
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
        for stem in dict.fromkeys(mod.RUNS):
            summary = self.rescored_files[stem]["summary"]
            stamp = summary.pop("rescored_with")
            with self.subTest(run=stem):
                text = self.exit_text()
                self.assertIn(f"{stem} was re-scored with None", text)
                self.assertIn(base.rescore.RESCORED_WITH, text)
                self.assert_stopped_before_the_build()
            summary["rescored_with"] = stamp

    def test_a_re_score_with_another_no_think_than_the_final_candidate_s_stops_main_naming_both_readings(self):
        for stem in [s for s in dict.fromkeys(mod.RUNS) if s != mod.RUNS[0]]:     # every run but the one compared with
            summary = self.rescored_files[stem]["summary"]
            summary["no_think"] = True
            with self.subTest(run=stem):
                text = self.exit_text()
                self.assertIn(f"{stem} was re-scored with no_think=True", text)          # the run and its reading
                self.assertIn("the final candidate with no_think=False: not one reading", text)
                self.assert_stopped_before_the_build()
            summary["no_think"] = False

    def test_a_final_candidate_that_alone_reads_another_no_think_stops_main_at_the_first_run_compared_with_it(self):
        # every run is compared with the first of RUNS: a final candidate that alone reads another way is caught at
        # the next run, stock, which the message names with both readings
        self.rescored_files[mod.RUNS[0]]["summary"]["no_think"] = True
        self.assertIn("p3c_stock was re-scored with no_think=False, the final candidate with no_think=True",
                      self.exit_text())
        self.assert_stopped_before_the_build()

    def test_a_re_score_whose_figures_are_not_its_run_file_s_stops_main_naming_the_run_the_figure_and_both_values(self):
        for stem in dict.fromkeys(mod.RUNS):
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
        for stem in dict.fromkeys(mod.RUNS):
            for files, name in ((self.rescored_files, f"{stem}_rescored.json"), (self.run_files, f"{stem}.json")):
                file = files.pop(stem)
                with self.subTest(run=stem, file=name):
                    # the reader's error becomes the message; there is no traceback; the hint names what is missing
                    self.assertEqual(self.exit_text(),
                                     f"{stem}: No such file or directory: {name} "
                                     + ("(re-score the run first)" if name.endswith("_rescored.json")
                                        else "(the run's files are not collected)"))
                    self.assert_stopped_before_the_build()
                files[stem] = file

    def test_a_stop_in_any_of_the_four_added_checks_leaves_no_endpoints_file_naming_that_run(self):
        # the four other versions get the checks too, after the seven runs: another checker, another no_think, or
        # figures that are not the run file's
        for stem, _ in OTHERS:
            summaries = dict(re_score=self.rescored_files[stem]["summary"], run=self.run_files[stem]["summary"])
            for kind, field, bad, good, words in FAILINGS:
                summaries[kind][field] = bad
                with self.subTest(run=stem, check=words):
                    text = self.exit_text()
                    self.assertIn(stem, text)                 # the run
                    self.assertIn(words, text)                # the check that stopped main()
                    self.assert_stopped_before_the_build()
                summaries[kind][field] = good

    # ---- the checks first, then the build ----
    def test_checks_every_run_of_runs_then_every_other_version_in_their_order_and_builds_only_then(self):
        self.run_main()
        self.assertEqual(self.log, [("checked", stem) for stem in mod.RUNS]
                         + [("checked", stem) for stem, _ in mod.OTHERS] + [("build", None)])

    def test_checks_the_seven_runs_of_the_report_and_the_four_other_versions_before_anything_is_built(self):
        # this phase's final candidate and its reference in the phase-14 script's order with stock, the published edit
        # and the two other reference arms, then the phase-14 editor's final; then the four other versions; each once
        self.run_main()
        self.assertEqual([stem for kind, stem in self.log if kind == "checked"],
                         ["p15_final", "p3c_stock", "p3c_c3_original", "p15_final_ref", "ref_stock", "ref_orig",
                          "p14_final", "p4_mimo", "p10_fullbudget", "p7_stack_full_v2", "p7_uperp_on_loop"])

    def test_the_phase_14_script_is_built_once_under_this_phases_names_and_folder(self):
        # the final candidate and its reference take this phase's names; stock, the published edit and the two other
        # reference arms keep theirs; the folder is this phase's
        self.run_main()
        self.built.assert_called_once_with()
        self.assertEqual(self.seen, [dict(self.at_start, FINAL="p15_final", REF_FINAL="p15_final_ref",
                                          HERE=self.folder15)])

    # ---- this phase's additions ----
    def test_exploratory_runs_holds_the_phase_14_editor_s_final_as_the_phase_14_script_describes_it(self):
        # the first of the five runs main() describes
        self.run_main()
        self.assertEqual(self.described.call_args_list[0], mock.call(PREVIOUS, VERDICTS[PREVIOUS]))
        self.assertEqual(self.written()["exploratory_runs"], {"phase-14 editor's final": LEVELS[PREVIOUS]})

    def test_other_exploratory_runs_hold_each_other_version_as_the_phase_14_script_describes_it_in_others_order(self):
        # after the phase-14 editor's final: {name: level}, the names in OTHERS order
        self.run_main()
        self.assertEqual(self.described.call_args_list[1:], [mock.call(stem, VERDICTS[stem]) for stem, _ in OTHERS])
        self.assertEqual(self.written()["other_exploratory_runs"], {name: LEVELS[stem] for stem, name in OTHERS})
        self.assertEqual(list(self.written()["other_exploratory_runs"]), [name for _, name in OTHERS])

    def test_the_exploratory_pair_pairs_the_verdicts_of_p15_final_with_those_of_p14_final_in_that_order(self):
        # {problem id: ok_rescored} over the rows of each re-score file: the first of the five pairs main() asks for
        self.run_main()
        self.assertEqual(len(self.paired), 1 + len(OTHERS))
        first, second, label, _ = self.paired[0]
        self.assertEqual((first, second), (VERDICTS[FINAL], VERDICTS[PREVIOUS]))
        self.assertIn("phase-14 editor's final", label)

    def test_the_other_four_pairs_pair_the_verdicts_of_p15_final_with_each_other_version_s_in_order(self):
        self.run_main()
        expected = [(VERDICTS[FINAL], VERDICTS[stem], f"final candidate against {name} (no protocol word)", None)
                    for stem, name in OTHERS]
        self.assertEqual(self.paired[1:], expected)

    def test_every_exploratory_pair_is_asked_for_without_a_protocol_word(self):
        # pair() takes the protocol's word as a fourth argument (word_key); none is given, five times
        self.run_main()
        self.assertEqual([word_key for *_, word_key in self.paired], [None] * (1 + len(OTHERS)))

    # ---- the one write, the one print ----
    def test_writes_the_output_once_itself_without_running_the_phase_14_script_s_main(self):
        self.run_main()
        self.wrote.assert_called_once_with(self.folder15 / "endpoints.json", mock.ANY)
        self.base_main.assert_not_called()
        self.assertEqual([p.name for p in self.folder15.iterdir()], ["endpoints.json"])    # no .tmp left beside it
        self.assertEqual([p.name for p in self.folder14.iterdir()], [])                     # phase14/: as it was

    def test_prints_what_it_wrote(self):
        self.run_main()
        self.assertEqual(self.printed, (self.folder15 / "endpoints.json").read_text(encoding="utf-8"))

    def test_the_endpoints_file_holds_the_phase_14_script_s_document_with_this_phase_s_additions(self):
        self.run_main()
        written = self.written()
        self.assertEqual(written["exploratory_runs"], {"phase-14 editor's final": LEVELS[PREVIOUS]})
        self.assertEqual(written["exploratory_pairs"], [PAIR])                    # as pair() returned it
        self.assertEqual(written["other_exploratory_pairs"], [PAIR] * len(OTHERS))   # as pair() returned them
        self.assertEqual(sorted(written), sorted([*BASE_DOCUMENT, *ADDED]))
        self.assertEqual({k: v for k, v in written.items() if k not in ADDED and k != "about"},
                         {k: v for k, v in BASE_DOCUMENT.items() if k not in ADDED and k != "about"})

    def test_the_about_text_is_the_phase_14_script_s_extended(self):
        self.run_main()
        expected = (BASE_DOCUMENT["about"] + " exploratory_runs, exploratory_pairs, other_exploratory_runs and "
                    "other_exploratory_pairs are this phase's additions and carry no protocol word.")
        self.assertEqual(self.written()["about"], expected)

    # ---- the names go back ----
    def test_puts_the_names_back_after_it_returns(self):
        self.run_main()
        self.assertEqual({name: getattr(base, name) for name in NAMES + ("HERE",)}, self.at_start)

    def test_puts_the_names_back_after_it_stops(self):
        self.rescored_files["ref_stock"]["summary"]["rescored_with"] = OTHER_STAMP
        with self.assertRaises(SystemExit):
            self.run_main()
        self.assertEqual({name: getattr(base, name) for name in NAMES + ("HERE",)}, self.at_start)

    # ---- a failure after the build leaves no endpoints file ----
    def test_a_refusal_inside_the_exploratory_pair_leaves_no_endpoints_file(self):
        def refusing(first, second, label, word_key=None):
            sys.exit(f"{label}: the two runs do not hold the same 500 problems")   # what the real pair() does then

        self.stand_in(base, "pair", refusing)
        with self.assertRaises(SystemExit):
            self.run_main()
        self.built.assert_called_once_with()             # the refusal came after the build
        self.assert_nothing_written()

    def test_a_phase_14_editor_s_final_that_is_not_a_measurement_leaves_no_endpoints_file(self):
        # the checks read no validity record; base.describe(), which exploratory_runs comes from, refuses the run
        self.rescored_files["p14_final"]["summary"]["validity"] = FAILED
        text = self.exit_text()
        self.assertEqual(text, f"p14_final is not a measurement: its validity record reads {FAILED} (ISSUES #20)")
        self.built.assert_called_once_with()
        self.assert_nothing_written()

    def test_an_error_after_the_build_leaves_no_endpoints_file(self):
        self.described.side_effect = OSError("the re-score file of p14_final cannot be read")
        with self.assertRaises(OSError):
            self.run_main()
        self.built.assert_called_once_with()
        self.assert_nothing_written()


class ImportTest(WithStandIns):
    """Importing phase15/compute_endpoints.py"""

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
