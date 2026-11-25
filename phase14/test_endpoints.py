"""Tests for phase14/endpoints.py, which computes the phase-14 endpoints from the collected files without a GPU
(phase14/protocol.md, "Endpoints" and "Verdict"; ISSUES #31 for the ways of choosing an answer line).
  word           the protocol's word for a paired difference with its 95% interval;
  repaired       REPAIR or not, decided on counts, and the bars as whole answers;
  last_box       the last one-level \\boxed{...} of a generation;
  readings       the checker's verdict, the section's last line read instead, and reading C, for one generation;
  math_verify_ok the second grader on a boxed answer;
  pair           paired statistics of two runs over the same 500 problems;
  describe       a run's level, from the two small JSON files it reads, for a run that is a measurement (ISSUES #20);
  write_atomic   a file replaced only once the new content is complete, with its line endings as given.
main() and the functions that read the collected runs are not covered. The generations are short stand-ins for real
ones; where a line comes from the record, its comment says so.

Standard library only, except for math_verify_ok() and pair(), which need the grader's environment: their tests
are skipped unless math-verify, numpy and phase8/analyze_saved.py import here, and so run inside WSL only.
Usage (from the repository root):
  python -B -m unittest -v phase14/test_endpoints.py
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the module under test is imported by bare name, as test_save_edit.py does
import endpoints  # noqa: E402
from endpoints import describe, last_box, math_verify_ok, pair, readings, repaired, word, write_atomic  # noqa: E402
from rescore import answers_match, extract_answer  # noqa: E402  (the checker endpoints imports; readings() is held to it)

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


class LastBoxTest(unittest.TestCase):
    """last_box(): the content of the last one-level \\boxed{...} of a generation; "" when there is none"""

    def test_returns_what_the_box_holds(self):
        self.assertEqual(last_box("The final answer is $\\boxed{42}$."), "42")

    def test_returns_the_last_box_of_the_generation(self):
        self.assertEqual(last_box("$\\boxed{3}$ is too small, so the answer is $\\boxed{7}$"), "7")

    def test_reads_a_box_with_one_brace_level_inside(self):
        self.assertEqual(last_box("The final answer is $\\boxed{\\dfrac{1}{2}}$."), "\\dfrac{1}{2}")

    def test_returns_an_empty_string_when_there_is_no_box(self):
        self.assertEqual(last_box("The final answer is 42."), "")

    def test_an_empty_box_counts_as_no_box(self):
        self.assertEqual(last_box("The final answer is $\\boxed{}$."), "")

    def test_a_box_nested_two_brace_levels_deep_is_not_read(self):
        # known limit of the checker's own extractor, which this is: such a generation counts as having no boxed
        # answer (ISSUES #31: an answer that contains \boxed{} is read the same under every rule)
        self.assertEqual(last_box("The final answer is $\\boxed{\\frac{\\sqrt{3}}{3}}$."), "")


def generation(answer_section):
    """A generation under the repository's question: one line of thinking, closed, then the answer section."""
    return "Let me work through this step by step.</think>\n" + answer_section


def checked(text, gold):
    """The checker's verdict on a generation: what a re-score file records as ok_rescored."""
    return answers_match(gold, extract_answer(text))


CONCLUSIONS = (  # the last line of an answer section; each states 4 in one of the shapes reading C accepts, and
    "The final answer is $4$.",             # only in that one: a $math$ span
    r"\[4\]",                               # a display block
    "The number of sides: **4**.",          # a bold value
    "So the number of sides is 4.",         # a linking word and a number
)


class ReadingsTest(unittest.TestCase):
    """readings(): (the checker's verdict, the section's last line read instead, reading C of ISSUES #31)"""

    # ---- read the same under every rule ----
    def test_a_boxed_answer_reads_the_same_under_every_rule_whatever_follows(self):
        # after the box: a stray fragment, which the last-line rule would read as "MATH"; or no answer section at all
        for text in (generation("The final answer is $\\boxed{4}$.\nIf a student is in MATH"),
                     "Still thinking: the best candidate so far is $\\boxed{4}$, but let me check"):
            with self.subTest(text=text):
                self.assertEqual(readings(text, "4", True), (True, True, True))

    def test_a_generation_whose_thinking_never_closed_reads_the_same_under_every_rule(self):
        text = "Still thinking: the reasoning was cut off before any answer"
        for ok in (True, False):
            with self.subTest(ok=ok):
                self.assertEqual(readings(text, "4", ok), (ok, ok, ok))

    # ---- reading C: the section's last line when it concludes, else the checker ----
    def test_a_concluding_last_line_is_read_right_whatever_verdict_the_generation_is_given(self):
        # the given verdict stands in the first place; the last line is read afresh in the other two
        text = generation("**Answer:**\nThe value of $x$ is **4**.")
        self.assertEqual(readings(text, "4", False), (False, True, True))

    def test_a_conclusion_on_the_last_line_supplies_what_the_checker_missed(self):
        # the phase-10 edit on problem m3 of the record (ISSUES #31): the checker took the heading that follows
        # "**Answer:**" for the answer, "196", and stopped there
        text = generation("**Answer:**\nTo find the number of positive whole-number divisors of $196$, we first find "
                          "the prime factorization.\nTotal number of divisors = $3 \\times 3 = 9$.\n"
                          "There are **9** positive whole-number divisors of 196.")
        self.assertEqual((checked(text, "9"), readings(text, "9", False)), (False, (False, True, True)))

    def test_a_conclusion_on_the_last_line_overrides_an_earlier_marker_line_under_reading_c(self):
        # the other side of ISSUES #31: the last line is trusted whenever it concludes, and turns wrong an answer
        # that the checker read right
        self.assertEqual(readings(generation("The answer is 4.\nSo the number of sides is 5."), "4", True),
                         (True, False, False))

    def test_each_of_the_four_concluding_shapes_makes_the_last_line_the_answer(self):
        for last in CONCLUSIONS:
            with self.subTest(last=last):
                self.assertEqual(readings(generation("**Answer:**\n" + last), "4", False), (False, True, True))

    def test_a_last_line_without_a_concluding_shape_falls_back_to_the_checker(self):
        # the end of a generation of the phase-10 edit (problem m59), which runs on for 2,500 lines after its answer
        text = generation("The answer is 5.\nWait.\nIf a student is in MATH")
        self.assertEqual((checked(text, "5"), readings(text, "5", True)), (True, (True, False, True)))

    def test_an_equals_sign_with_spaces_around_it_is_not_a_concluding_shape(self):
        # known limit, noted where STRUCT is defined: the word boundary before the linking word stands before "="
        # too, so on a last line like this reading C falls back to the checker
        self.assertEqual(readings(generation("So the number of sides is 4."), "4", False), (False, True, True))
        self.assertEqual(readings(generation("So the number of sides = 4."), "4", False), (False, True, False))

    # ---- the checker's treatment of the section is kept ----
    def test_an_answer_section_of_white_space_only_reads_wrong_under_the_last_line_rule(self):
        for text in ("Cut off here</think>\n  \n\t", "Cut off here</think>"):
            for ok in (True, False):
                with self.subTest(text=text, ok=ok):
                    self.assertEqual(readings(text, "4", ok), (ok, False, ok))

    def test_a_stray_think_token_in_the_section_is_dropped_as_the_checker_drops_it(self):
        # ISSUES #19 c: in stop-token-boosted arms the model emits </think> again inside the answer section. Read
        # as text, the token would turn the last line into a LaTeX payload
        for section, gold in (("The answer is $4$.\n</think>", "4"),     # on a line of its own after the answer
                              ("</think> 42", "42"),                     # opening the last line
                              ("So x is </think> 5.", "5")):             # inside the last line
            text = generation(section)
            with self.subTest(section=section):
                self.assertEqual((checked(text, gold), readings(text, gold, True)), (True, (True, True, True)))


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


if __name__ == "__main__":
    unittest.main()
