"""Tests for phase14/endpoints.py, which computes the phase-14 endpoints from the collected files without a GPU
(phase14/protocol.md, "Endpoints" and "Verdict").
  word            the protocol's word for a paired difference with its 95% interval;
  repaired        REPAIR or not, decided on counts, and the bars as whole answers;
  boxed_readings  the two graders on each generation's last boxed answer for one run, read as the checker reads it;
  math_verify_ok  the second grader on a boxed answer;
  pair            paired statistics of two runs over the same 500 problems;
  describe        a run's level, from the two small JSON files it reads, for a run that is a measurement (ISSUES #20);
  same_checker    main()'s guard: stop unless the six re-scored files were made by the checker in this tree;
  grader_versions the record of the two graders, with the stamp of that checker;
  write_atomic    a file replaced only once the new content is complete, with its line endings as given;
  main            what build() returns, written as it is into this folder's endpoints.json.
build() and the small readers rescored(), run() and texts() are not covered; main()'s test stands build() and
write_atomic() in, so that it reads no run file and writes nothing. The generations are short stand-ins for real
ones; where a line comes from the record, its comment says so.

Standard library only, except for math_verify_ok() and pair(), which need the grader's environment: their tests
are skipped unless math-verify, numpy and phase8/analyze_saved.py import here, and so run inside WSL only.
boxed_readings() needs math-verify too; its test stands the second grader in, so that it runs here as well and
says the same in both environments.
Usage (from the repository root):
  python -B -m unittest -v phase14/test_endpoints.py
"""

import contextlib
import gzip
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the module under test is imported by bare name, as test_save_edit.py does
import endpoints  # noqa: E402
from endpoints import (boxed_readings, describe, grader_versions, math_verify_ok, pair, repaired, same_checker,  # noqa: E402
                     word, write_atomic)

sys.path.insert(0, str(HERE.parent / "phase8"))  # where pair() imports analyze_saved from


def importable(*names):
    """Whether every module in `names` imports here."""
    try:
        for name in names:
            importlib.import_module(name)
    except ImportError:
        return False
    return True


HAVE_MATH_VERIFY = importable("math_verify", "math_verify.errors")
HAVE_PAIRED = importable("numpy", "analyze_saved")   # what pair() imports; analyze_saved needs scipy and math-verify


class WordTest(unittest.TestCase):
    """word(): the protocol's word for (a paired difference, the lower and the upper end of its 95% interval)"""

    # ---- within the band, edges included ----
    def test_no_difference_reaches_it(self):
        self.assertEqual(word(0.0, -3.0, 3.0), "REACHES IT")

    def test_four_points_above_exactly_reaches_it(self):
        # on the band's edge, although the interval excludes zero
        self.assertEqual(word(4.0, 0.2, 7.8), "REACHES IT")

    def test_four_points_below_exactly_reaches_it(self):
        # B3 against B1 as first read (ISSUES #31): a hit exactly on its bar
        self.assertEqual(word(-4.0, -8.6, 0.6), "REACHES IT")

    # ---- below the band ----
    def test_more_than_four_below_with_the_interval_below_zero_falls_short(self):
        # the final candidate against the published edit (phase14/endpoints.json)
        self.assertEqual(word(-39.6, -44.0, -35.2), "FALLS SHORT")

    def test_more_than_four_below_with_the_interval_across_zero_is_inconclusive(self):
        # the same comparison under the other two readings of ISSUES #31: 0.2 points outside the band
        self.assertEqual(word(-4.2, -8.6, 0.4), "INCONCLUSIVE")

    def test_more_than_four_below_with_the_interval_ending_at_zero_is_inconclusive(self):
        self.assertEqual(word(-4.2, -8.6, 0.0), "INCONCLUSIVE")

    # ---- above the band ----
    def test_more_than_four_above_with_the_interval_above_zero_exceeds(self):
        self.assertEqual(word(7.2, 2.8, 11.8), "EXCEEDS")

    def test_more_than_four_above_with_the_interval_across_zero_is_inconclusive(self):
        self.assertEqual(word(7.2, -0.4, 11.8), "INCONCLUSIVE")

    def test_more_than_four_above_with_the_interval_starting_at_zero_is_inconclusive(self):
        self.assertEqual(word(7.2, 0.0, 11.8), "INCONCLUSIVE")


class RepairedTest(unittest.TestCase):
    """repaired(): REPAIR or not on counts, from (the final's right answers, stock's, the number of problems)"""

    # ---- decided on counts, not on rounded percentages ----
    def test_holds_exactly_at_both_bars(self):
        # 80 of 500 is 16.0%, and 50 answers more than stock's 30 is 10.0 points
        self.assertTrue(repaired(80, 30, 500)["holds"])

    def test_one_answer_below_the_percentage_bar_does_not_hold(self):
        self.assertFalse(repaired(79, 29, 500)["holds"])     # the gap is still 50 answers

    def test_one_answer_below_the_gap_bar_does_not_hold(self):
        self.assertFalse(repaired(80, 31, 500)["holds"])     # the level is still 16.0%

    def test_an_exact_gap_of_ten_points_holds_when_the_rounded_percentages_disagree(self):
        # 92 of 500 against stock's 42 of 500: 18.4% against 8.4%, exactly 50 answers apart
        self.assertLess(18.4 - 8.4, 10.0)                    # a decision on the rounded percentages would fail here
        self.assertTrue(repaired(92, 42, 500)["holds"])

    def test_holds_well_above_both_bars(self):
        self.assertTrue(repaired(100, 40, 500)["holds"])

    def test_nothing_right_does_not_hold(self):
        self.assertFalse(repaired(0, 0, 500)["holds"])

    def test_a_run_with_no_problems_does_not_hold(self):
        # nothing measured: both bars are cleared by zero, and still it is not a repair
        self.assertFalse(repaired(0, 0, 0)["holds"])

    # ---- what it reports ----
    def test_reports_both_bars_as_counts_of_the_problems(self):
        out = repaired(45, 42, 500)
        self.assertEqual((out["min_right"], out["min_gap"]), (80, 50))

    def test_reports_the_bars_rounded_up_to_whole_answers(self):
        # 16.0% of 40 problems is 6.4 and 10.0 points of 40 is 4.0
        out = repaired(0, 0, 40)
        self.assertEqual((out["min_right"], out["min_gap"]), (7, 4))
        # 10.0 points of 45 is 4.5
        self.assertEqual(repaired(0, 0, 45)["min_gap"], 5)

    def test_six_right_of_40_does_not_hold_and_seven_does(self):
        # the decision turns at the bar reported: 15.0%, then 17.5% and 7.0 points above stock
        self.assertFalse(repaired(6, 0, 40)["holds"])
        self.assertTrue(repaired(7, 0, 40)["holds"])

    def test_a_gap_of_four_answers_of_45_does_not_hold_and_five_does(self):
        # 8.9 points above stock, then 11.1
        self.assertFalse(repaired(12, 8, 45)["holds"])
        self.assertTrue(repaired(13, 8, 45)["holds"])

    def test_records_the_counts_it_was_decided_on(self):
        out = repaired(45, 42, 500)
        self.assertEqual((out["right_final"], out["right_stock"], out["n"], out["holds"]), (45, 42, 500, False))


def generation(answer_section):
    """A generation under the repository's question: one line of thinking, closed, then the answer section."""
    return "Let me work through this step by step.</think>\n" + answer_section


class BoxedReadingsTest(unittest.TestCase):
    """boxed_readings(): (the repository's matcher, math-verify) on the last boxed answer of each generation"""

    def read(self, texts, golds):
        """boxed_readings("x") with a temporary folder standing in for the script's P3, for a run called "x" whose
        generations are `texts` and whose rows carry the gold answers `golds` ({problem id: text or gold}); with
        them, the boxed answers the second grader was given. math_verify_ok() is stood in: the real one raises
        where math-verify is not installed."""
        with tempfile.TemporaryDirectory() as tmp:
            p3 = Path(tmp)
            (p3 / "x_rescored.json").write_text(json.dumps(dict(rows=[dict(id=i, gold=g) for i, g in golds.items()])),
                                                encoding="utf-8")
            with gzip.open(p3 / "x_texts.json.gz", "wt", encoding="utf-8") as f:
                json.dump(texts, f)
            with mock.patch.object(endpoints, "P3", p3), mock.patch.object(
                    endpoints, "math_verify_ok", side_effect=lambda gold, boxed: gold == boxed) as second:
                return boxed_readings("x"), [call.args[1] for call in second.call_args_list]

    def test_a_box_nested_two_brace_levels_deep_is_read(self):
        # since 2026-11-26 the last box is read to its matching brace at any depth (ISSUES #30). The one-level
        # pattern this replaces missed this box: the answer counted as not boxed, and was wrong
        texts = {"m0": generation("The final answer is $\\boxed{\\frac{\\sqrt{3}}{3}}$."),
                 "m1": generation("The final answer is $\\boxed{41}$."),
                 "m2": generation("The final answer is 41.")}
        golds = {"m0": "\\frac{\\sqrt{3}}{3}", "m1": "42", "m2": "42"}
        (out, attempted, unreadable), given = self.read(texts, golds)
        self.assertEqual(out, {"m0": (True, True), "m1": (False, False), "m2": (False, False)})
        self.assertEqual(given, ["\\frac{\\sqrt{3}}{3}", "41"])     # the second grader gets each box read whole
        self.assertEqual((attempted, unreadable), (2, 0))           # the two generations that hold a box


@unittest.skipUnless(HAVE_MATH_VERIFY, "math_verify_ok() needs math-verify, which is not installed here")
class MathVerifyOkTest(unittest.TestCase):
    """math_verify_ok(): the second grader on (gold, boxed answer)"""

    def test_a_right_answer_is_true(self):
        self.assertIs(math_verify_ok("4", "4"), True)

    def test_a_wrong_answer_is_false(self):
        # False, not None: None is an answer math-verify cannot read, which is counted as ungraded
        self.assertIs(math_verify_ok("4", "5"), False)


IDS = [f"m{i}" for i in range(500)]   # the problem ids of a MATH-500 run


def verdicts(right, ids=IDS):
    """{problem id: verdict} with one entry per id in `ids` (the 500 problems unless said otherwise), right for the
    ids in `right`, entered in the order of `ids`."""
    right = set(right)
    return {i: i in right for i in ids}


@unittest.skipUnless(HAVE_PAIRED, "pair() needs numpy and phase8/analyze_saved.py, which do not import here")
class PairTest(unittest.TestCase):
    """pair(): paired statistics of two runs over the same 500 problems"""

    # ---- refused ----
    def test_two_runs_that_do_not_hold_the_same_500_problems_are_refused(self):
        # sys.exit(<text>): the text is the exit object here and goes to stderr with status 1 on the real command line
        label = "final candidate against stock"
        runs = dict(first_run_short=(IDS[1:], IDS), second_run_short=(IDS, IDS[1:]),
                    second_run_long=(IDS, IDS + ["m500"]), another_problem=(IDS, IDS[:-1] + ["m500"]))
        for name, (first, second) in runs.items():
            with self.subTest(case=name), self.assertRaises(SystemExit) as raised:
                pair(verdicts(IDS[:100], first), verdicts(IDS[:100], second), label)
            self.assertEqual(raised.exception.code, f"{label}: the two runs do not hold the same 500 problems")

    # ---- paired by problem id ----
    def test_pairs_the_runs_by_problem_id_whatever_the_order_of_the_dicts(self):
        # the same 100 problems right in both runs, the second dict entered in reverse id order
        out = pair(verdicts(IDS[:100]), verdicts(IDS[:100], IDS[::-1]), "a against b")
        self.assertEqual((out["first"], out["second"], out["delta"], out["only_first"], out["only_second"]),
                         (20.0, 20.0, 0.0, 0, 0))
        self.assertEqual(out["p"], 1.0)     # no problem separates the two runs

    def test_counts_the_problems_that_only_one_run_got_right(self):
        # 100 right in each run, the same 50 of them: equal levels, different runs
        out = pair(verdicts(IDS[:100]), verdicts(IDS[50:150]), "a against b")
        self.assertEqual((out["only_first"], out["only_second"], out["first"], out["second"], out["delta"]),
                         (50, 50, 20.0, 20.0, 0.0))
        self.assertLess(out["ci95"][0], 0.0)
        self.assertGreater(out["ci95"][1], 0.0)

    def test_returns_the_two_levels_in_percent_and_first_minus_second_in_points(self):
        out = pair(verdicts(IDS[:100]), verdicts(IDS[:80]), "a against b")
        self.assertEqual((out["first"], out["second"], out["delta"], out["only_first"], out["only_second"]),
                         (20.0, 16.0, 4.0, 20, 0))

    def test_a_difference_of_exactly_four_points_reaches_it_though_the_interval_excludes_zero(self):
        # the band is inclusive, and it is asked first
        out = pair(verdicts(IDS[:100]), verdicts(IDS[:80]), "a against the published edit", "word")
        self.assertGreater(out["ci95"][0], 0.0)
        self.assertEqual(out["word"], "REACHES IT")

    def test_a_difference_beyond_the_band_whose_interval_includes_zero_is_inconclusive(self):
        # 90 problems right only under the first run, 69 only under the second: +4.2 points, the interval over zero
        out = pair(verdicts(IDS[:90]), verdicts(IDS[90:159]), "a against the published edit", "word")
        self.assertEqual((out["delta"], out["only_first"], out["only_second"]), (4.2, 90, 69))
        self.assertLess(out["ci95"][0], 0.0)
        self.assertGreater(out["ci95"][1], 0.0)
        self.assertEqual(out["word"], "INCONCLUSIVE")

    def test_puts_the_word_exceeds_under_the_word_key_when_the_interval_lies_above_zero(self):
        out = pair(verdicts(IDS[:100]), verdicts(IDS[:50]), "a against the published edit", "word")
        self.assertEqual((out["delta"], out["only_first"], out["only_second"]), (10.0, 50, 0))
        self.assertGreater(out["ci95"][0], 0.0)
        self.assertEqual(out["word"], "EXCEEDS")
        self.assertLess(out["p"], 1e-10)     # the exact test on 50 problems against none

    def test_puts_the_word_falls_short_under_the_word_key_for_the_same_comparison_the_other_way_round(self):
        out = pair(verdicts(IDS[:50]), verdicts(IDS[:100]), "a against the published edit", "word")
        self.assertEqual((out["delta"], out["only_first"], out["only_second"]), (-10.0, 0, 50))
        self.assertLess(out["ci95"][1], 0.0)
        self.assertEqual(out["word"], "FALLS SHORT")

    def test_adds_no_word_without_a_word_key(self):
        # the protocol has a word for the comparison with the published edit only
        out = pair(verdicts(IDS), verdicts([]), "a against stock")
        self.assertEqual((out["label"], out["n"]), ("a against stock", 500))
        self.assertEqual(sorted(out), ["ci95", "delta", "first", "label", "n", "only_first", "only_second", "p",
                                       "second"])


STEM = "x"
STAMP = "phase3c/rescore.py+phase3b/scorer.py#4a1b56a343a4 (ISSUES #19)"
CLEAN = dict(soup_pct=0.0, soup_judged=True, admission_tested=True, problems=[])
FAILED = dict(CLEAN, problems=["soup: 30.0% of generations end in non-Latin script (bar 20.0%)"])
RUN_SUMMARY = dict(name=STEM, n=6, cap_hit_rate=25.0, mean_tokens=812.5, wall_s=63.0)
# a run's problems take turns in these two six-row patterns: the thinking closed in the first of the six, and the
# verdicts given to describe() right in four of the six
CLOSED = (True, False, False, False, False, False)
RIGHT = (True, True, True, False, True, False)


def without(record, key):
    """The validity record `record` with the field `key` left out: a record that does not say."""
    return {k: v for k, v in record.items() if k != key}


def not_a_measurement(validity):
    """The text describe() exits with: the run, the validity record it read, and the issue."""
    return f"{STEM} is not a measurement: its validity record reads {validity} (ISSUES #20)"


def write_run(p3, n=6, validity=CLEAN, run_summary=None, rescore_summary=None):
    """In `p3`, the two files describe() reads for a run called "x" of `n` problems: x.json, the run, whose summary
    is `run_summary` (by default RUN_SUMMARY), and x_rescored.json, its re-score with `rescore_summary` (by default
    the checker's stamp and the validity record `validity`). Returns the verdicts, {problem id: verdict}."""
    ids = [f"m{i}" for i in range(n)]
    rows = [dict(id=i, closed=CLOSED[k % len(CLOSED)]) for k, i in enumerate(ids)]
    if run_summary is None:
        run_summary = dict(RUN_SUMMARY, n=n)
    if rescore_summary is None:
        rescore_summary = dict(name="x_rescored", n=n, rescored_with=STAMP, validity=validity)
    (p3 / f"{STEM}.json").write_text(json.dumps(dict(summary=run_summary, rows=rows)), encoding="utf-8")
    (p3 / f"{STEM}_rescored.json").write_text(json.dumps(dict(summary=rescore_summary, rows=rows)), encoding="utf-8")
    return {i: RIGHT[k % len(RIGHT)] for k, i in enumerate(ids)}


class DescribeTest(unittest.TestCase):
    """describe(): a run's level from the verdicts given, its re-score file and its run file"""

    def described(self, **files):
        """describe(STEM, the run's verdicts) with a temporary folder standing in for the script's P3; `files` go to
        write_run()."""
        with tempfile.TemporaryDirectory() as tmp:
            p3 = Path(tmp)
            right = write_run(p3, **files)
            with mock.patch.object(endpoints, "P3", p3):
                return describe(STEM, right)

    def refused(self, **files):
        """The text describe() exits with on the run that `files` describe."""
        with self.assertRaises(SystemExit) as raised:
            self.described(**files)
        return raised.exception.code

    def test_a_clean_run_is_described_from_its_verdicts_its_rows_and_its_summaries(self):
        # accuracy and the count from the verdicts given; closed from the re-score's rows; cap_hit_rate,
        # mean_tokens and wall_s from the run; validity and the stamp from the re-score's summary. Four of six
        # and one of six: both shares come out with a decimal
        self.assertEqual(self.described(), dict(accuracy=66.7, right=4, n=6, cap_hit_rate=25.0, closed=16.7,
                                                mean_tokens=812.5, wall_s=63.0, validity=CLEAN,
                                                rescore_file_stamp=STAMP))

    def test_a_run_that_records_no_cap_hit_rate_is_described_with_none(self):
        # runs made by the phase-8 runner carry none
        summary = {k: v for k, v in RUN_SUMMARY.items() if k != "cap_hit_rate"}
        self.assertIs(self.described(run_summary=summary)["cap_hit_rate"], None)

    # ---- refused: not a measurement (ISSUES #20) ----
    def test_a_re_score_whose_validity_record_lists_problems_is_refused(self):
        summary = dict(name="x_rescored", n=6, rescored_with=STAMP, validity=FAILED)
        with self.assertRaises(SystemExit) as raised:
            self.described(rescore_summary=summary)
        self.assertEqual(raised.exception.code,
                         f"{STEM} is not a measurement: its validity record reads {FAILED} (ISSUES #20)")

    def test_a_re_score_without_a_validity_record_is_refused(self):
        for validity in (None, {}):      # None: the summary carries no validity record at all
            summary = dict(name="x_rescored", n=6, rescored_with=STAMP)
            if validity is not None:
                summary["validity"] = validity
            with self.subTest(validity=validity), self.assertRaises(SystemExit) as raised:
                self.described(rescore_summary=summary)
            self.assertEqual(raised.exception.code,
                             f"{STEM} is not a measurement: its validity record reads {validity} (ISSUES #20)")

    # ---- refused: a check that did not run, or that the record does not mention ----
    def test_a_record_that_does_not_record_a_judged_fragment_check_is_refused(self):
        for record in (dict(CLEAN, soup_judged=False), without(CLEAN, "soup_judged")):
            with self.subTest(record=record):
                self.assertEqual(self.refused(validity=record), not_a_measurement(record))

    def test_a_run_of_100_rows_or_more_whose_admission_check_did_not_run_is_refused(self):
        # phase3c/runcheck.py gives every run of that many rows (ADMISSION_FLOOR) the admission-order check
        not_given = dict(CLEAN, admission_tested=False)
        for n, record in ((100, not_given), (500, not_given), (100, without(CLEAN, "admission_tested"))):
            with self.subTest(n=n, record=record):
                self.assertEqual(self.refused(n=n, validity=record), not_a_measurement(record))

    def test_a_record_whose_problems_are_not_an_empty_list_is_refused(self):
        for record in (dict(CLEAN, problems=None), without(CLEAN, "problems")):
            with self.subTest(record=record):
                self.assertEqual(self.refused(validity=record), not_a_measurement(record))

    def test_a_validity_record_that_is_not_a_dict_is_refused(self):
        problem = FAILED["problems"][0]
        for record in (None, [problem], "clean"):     # JSON null, a list of problems, a string
            with self.subTest(record=record):
                self.assertEqual(self.refused(validity=record), not_a_measurement(record))

    # ---- accepted ----
    def test_a_run_of_fewer_than_100_rows_need_not_have_been_given_the_admission_check(self):
        # runcheck.py gives no run of fewer than 100 rows that check, and its record says so
        out = self.described(n=99, validity=dict(CLEAN, admission_tested=False))
        self.assertEqual((out["n"], out["right"], out["accuracy"]), (99, 67, 67.7))     # 67 of the 99 right

    def test_a_run_of_100_rows_with_every_check_on_record_is_a_measurement(self):
        out = self.described(n=100)
        self.assertEqual((out["n"], out["right"], out["accuracy"]), (100, 67, 67.0))    # 67 of the 100 right


CHECKED = ("p14_final", "p3c_stock", "p3c_c3_original", "p14_final_ref", "ref_stock", "ref_orig")   # the guard's six runs


class SameCheckerTest(unittest.TestCase):
    """same_checker(): main()'s guard, which stops unless the six re-scored files were made by this tree's checker"""

    def guarded(self, stamps):
        """same_checker() with a temporary folder standing in for the script's P3, holding <run>_rescored.json for
        each {run: the checker its summary records} in `stamps` (None: a summary that records none)."""
        with tempfile.TemporaryDirectory() as tmp:
            p3 = Path(tmp)
            for run, stamp in stamps.items():
                summary = dict(name=f"{run}_rescored") if stamp is None else dict(name=f"{run}_rescored", rescored_with=stamp)
                (p3 / f"{run}_rescored.json").write_text(json.dumps(dict(summary=summary, rows=[])), encoding="utf-8")
            with mock.patch.object(endpoints, "P3", p3):
                return same_checker()

    def stamped(self, **other):
        """The six runs with this tree's checker, except those in `other` ({run: what re-scored them instead})."""
        return {run: other.get(run, endpoints.rescore.RESCORED_WITH) for run in CHECKED}

    def test_passes_when_the_six_re_scored_files_carry_this_trees_checker(self):
        self.assertIsNone(self.guarded(self.stamped()))

    def test_stops_when_any_of_the_six_carries_another_checker(self):
        self.assertNotEqual(STAMP, endpoints.rescore.RESCORED_WITH)      # STAMP is the frozen checker of the record
        for run in CHECKED:
            with self.subTest(run=run), self.assertRaises(SystemExit) as raised:
                self.guarded(self.stamped(**{run: STAMP}))
            message = str(raised.exception.code)
            self.assertIn(run, message)                                  # the run
            self.assertIn(STAMP, message)                                # the checker it was re-scored with
            self.assertIn(endpoints.rescore.RESCORED_WITH, message)      # the checker in this tree

    def test_stops_when_a_re_scored_file_records_no_checker(self):
        with self.assertRaises(SystemExit) as raised:
            self.guarded(self.stamped(p3c_stock=None))
        self.assertIn("p3c_stock", str(raised.exception.code))


class GraderVersionsTest(unittest.TestCase):
    """grader_versions(): the record of the graders a report was made with"""

    def test_records_the_checker_in_this_tree(self):
        self.assertEqual(grader_versions()["checker"], endpoints.rescore.RESCORED_WITH)

    def test_still_records_the_matcher_and_the_second_grader(self):
        out = grader_versions()
        self.assertEqual(sorted(out), ["checker", "matcher", "math_verify", "problems_in_sorted_order"])
        self.assertTrue(out["math_verify"].startswith("math-verify"), out)


class WriteAtomicTest(unittest.TestCase):
    """write_atomic(): a file replaced only once the new content is complete on disk"""

    def test_writes_the_text_and_leaves_no_temporary_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoints.json"
            write_atomic(path, '{"word": "REACHES IT"}\n')
            self.assertEqual(path.read_text(encoding="utf-8"), '{"word": "REACHES IT"}\n')
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["endpoints.json"])

    def test_replaces_an_older_file_and_leaves_no_temporary_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoints.json"
            path.write_text("the endpoints of an earlier run", encoding="utf-8")
            write_atomic(path, "the endpoints of this run")
            self.assertEqual(path.read_text(encoding="utf-8"), "the endpoints of this run")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["endpoints.json"])

    def test_leaves_the_older_file_as_it_was_when_the_new_content_cannot_be_written(self):
        # a folder sits where the temporary file would go
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoints.json"
            path.write_text("the endpoints of an earlier run", encoding="utf-8")
            path.with_name(path.name + ".tmp").mkdir()
            with self.assertRaises(OSError):
                write_atomic(path, "the endpoints of this run")
            self.assertEqual(path.read_text(encoding="utf-8"), "the endpoints of an earlier run")

    def test_writes_the_line_endings_as_given(self):
        # read as bytes: read_text() would translate any carriage returns away
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoints.json"
            write_atomic(path, "a\nb\n")
            self.assertEqual(path.read_bytes(), b"a\nb\n")


class MainTest(unittest.TestCase):
    """main(): the endpoints build() returns, written as they are"""

    def test_writes_exactly_what_build_returns_into_this_folders_endpoints_file(self):
        # build() and write_atomic() are stood in: no run file is read, and the stand-in is the only writer. The
        # document carries a text outside ASCII, as a validity record can (a fragment of a generation): what is
        # written must keep it as it is
        document = {"about": "rescore_file_stamp identifies the checker that produced each re-score file",
                    "validity": {"problems": ["soup: «réponse» ends the generation in French"]},
                    "primary_runs": {"final candidate": {"accuracy": 60.0, "n": 500}}}
        path = HERE / "endpoints.json"
        before = path.read_bytes() if path.exists() else None     # the file as it is on this checkout
        # what main() prints is kept out of the test run's output, and from a console that may not encode it
        with mock.patch.object(endpoints, "build", return_value=document) as build, mock.patch.object(
                endpoints, "write_atomic") as write_atomic, contextlib.redirect_stdout(io.StringIO()):
            endpoints.main()
        build.assert_called_once_with()
        write_atomic.assert_called_once_with(HERE / "endpoints.json",
                                             json.dumps(document, indent=1, ensure_ascii=False) + "\n")
        # nothing was written on the repository's disk: the file is as it was, or still absent
        self.assertEqual(path.read_bytes() if path.exists() else None, before)


if __name__ == "__main__":
    unittest.main()
