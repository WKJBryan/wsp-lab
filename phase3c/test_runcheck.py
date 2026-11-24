"""Tests for phase3c/runcheck.py, the validity checks on a finished generation run
(see the runcheck module docstring: a run that fails a check is not a measurement).
  soup_rate        percent of generations whose last 1500 characters contain a non-Latin script;
  fisher_p         two-sided exact Fisher test for a 2x2 table;
  admission_split  one outcome, first half of the rows against the second half;
  check_run        both checks over a whole run -> (problems, metrics);
  save_run         a finished run and its generations written, an existing file refused by refuse_existing;
  main             the command line over a saved run: sys.argv set, the exit status caught.
The two committed phase-7 `stack` finals close the file (skipped when their files are absent).
load_texts() is covered only through main().

Standard library only. Usage (from the repository root):
  python -B -m unittest -v phase3c/test_runcheck.py
"""

import contextlib
import gzip
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the module under test is imported by bare name, as test_rescore.py does
from runcheck import admission_split, check_run, fisher_p, main, refuse_existing, save_run, soup_rate  # noqa: E402

PLAIN = "The answer is 7."
SOUP_TAIL = "\u4e2d" * 30    # a fragment tail: thirty copies of one ideograph
EXAMPLE_TEXTS = {"m0": PLAIN, "m1": SOUP_TAIL}
EXAMPLE_RESULT = {"summary": {"name": "caf\u00e9 run", "accuracy": 50.0}, "rows": [{"id": "m0", "ok": True}]}


def stopped(row):
    """The outcome check_run itself tests as "finished"."""
    return row["finish"] == "stop"


def mentioning(problems, text):
    """The problems whose message contains `text`."""
    return [p for p in problems if text in p]


def rows_by_position(n, n_stop):
    """`n` rows m0..., in submission order: the first `n_stop` finished, the rest hit the length cap."""
    return [{"id": f"m{i}", "finish": "stop" if i < n_stop else "length"} for i in range(n)]


def full_run(ok=lambda i: i % 2 == 0):
    """500 rows, all finished with thinking closed; `ok(i)` is problem i's rescored verdict."""
    return [{"id": f"m{i}", "finish": "stop", "closed": True, "ok_rescored": ok(i)} for i in range(500)]


def texts_for(rows, soupy=lambda i: False):
    """A plain-English generation per row; the ones with `soupy(i)` end in a run of CJK characters."""
    return {r["id"]: (PLAIN + SOUP_TAIL if soupy(i) else PLAIN) for i, r in enumerate(rows)}


class SoupRateTest(unittest.TestCase):
    """soup_rate(): percent of generations whose tail carries a non-Latin script"""

    def test_all_english_tails_give_zero(self):
        self.assertEqual(soup_rate({"a": "The answer is 7.", "b": "Still thinking"}, ["a", "b"]), 0.0)

    def test_one_cjk_tail_of_two_gives_50_percent(self):
        self.assertEqual(soup_rate({"a": "ok", "b": "\u2026\u672c\u4e0c"}, ["a", "b"]), 50.0)

    def test_only_the_last_1500_characters_count(self):
        # U+4E2D is the 2001st character before the end: outside the tail, not counted
        self.assertEqual(soup_rate({"t": "\u4e2d" + "x" * 2000}, ["t"]), 0.0)
        # ... while the 1500th character before the end is still inside it
        self.assertEqual(soup_rate({"t": "\u4e2d" + "x" * 1499}, ["t"]), 100.0)

    def test_the_rate_is_a_percentage_of_every_id_given(self):
        self.assertAlmostEqual(soup_rate({"a": "ok", "b": "\u4e2d", "c": "ok"}, ["a", "b", "c"]), 33.333, places=3)


class FisherPTest(unittest.TestCase):
    """fisher_p(): two-sided exact Fisher test on the table [[a, b], [c, d]]"""

    def test_equal_rates_give_p_of_one(self):
        self.assertEqual(fisher_p(10, 10, 10, 10), 1.0)

    def test_a_perfect_split_is_tiny_but_above_zero(self):
        p = fisher_p(20, 0, 0, 20)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1e-6)

    def test_the_result_is_never_above_one(self):
        self.assertEqual(fisher_p(0, 0, 0, 0), 1.0)


class AdmissionSplitTest(unittest.TestCase):
    """admission_split(): (first-half rate %, second-half rate %, two-sided p) for one outcome"""

    def test_identical_halves_give_equal_rates_and_p_of_one(self):
        self.assertEqual(admission_split(rows_by_position(200, 200), stopped), (100.0, 100.0, 1.0))

    def test_an_outcome_that_depends_on_position_is_detected(self):
        first, second, p = admission_split(rows_by_position(200, 100), stopped)
        self.assertEqual((first, second), (100.0, 0.0))
        self.assertLess(p, 0.001)

    def test_rows_are_read_in_list_order_not_sorted_by_id(self):
        # the same 100 "stop" then 100 "length" rows, listed m199 first and m0 last
        rows = [{"id": f"m{199 - i}", "finish": "stop" if i < 100 else "length"} for i in range(200)]
        first, second, _ = admission_split(rows, stopped)
        self.assertEqual((first, second), (100.0, 0.0))

    def test_an_odd_row_count_puts_the_extra_row_in_the_second_half(self):
        # only the fifth row, the odd one out, finished: the halves are the first 2 and the last 3 rows
        rows = [{"id": f"m{i}", "finish": "stop" if i == 4 else "length"} for i in range(5)]
        first, second, p = admission_split(rows, stopped)
        self.assertAlmostEqual(first, 0.0, places=3)
        self.assertAlmostEqual(second, 100.0 / 3, places=3)
        self.assertTrue(0.0 <= p <= 1.0)


class CheckRunTest(unittest.TestCase):
    """check_run(): (problems, metrics) for a whole run, its rows in submission order"""

    def test_a_clean_run_passes(self):
        rows = full_run()
        problems, metrics = check_run(rows, texts_for(rows), ok_key="ok_rescored")
        self.assertEqual(problems, [])
        self.assertEqual(metrics["soup_pct"], 0.0)

    def test_fragment_tails_fail_the_soup_check(self):
        rows = full_run()
        problems, _ = check_run(rows, texts_for(rows, lambda i: i % 10 < 3), ok_key="ok_rescored")   # 30% of 500
        self.assertEqual(len(problems), 1)
        self.assertIn("soup", problems[0])

    def test_correctness_that_depends_on_batch_position_fails_the_admission_check(self):
        rows = full_run(lambda i: i < 250)
        problems, _ = check_run(rows, texts_for(rows), ok_key="ok_rescored")
        self.assertTrue(mentioning(problems, "admission order (correct)"), problems)

    def test_ok_key_none_leaves_correctness_untested(self):
        rows = full_run(lambda i: i < 250)
        problems, _ = check_run(rows, texts_for(rows), ok_key=None)
        self.assertEqual(problems, [])

    def test_ok_key_naming_a_missing_field_is_ignored(self):
        rows = full_run(lambda i: i < 250)
        problems, _ = check_run(rows, texts_for(rows), ok_key="no_such_field")
        self.assertEqual(problems, [])

    def test_too_few_rows_means_no_admission_test_at_all(self):
        rows = rows_by_position(99, 50)
        problems, metrics = check_run(rows, texts_for(rows))
        self.assertEqual(problems, [])
        # the soup check ran and was judged; the admission test did not run and says so
        self.assertEqual(metrics, {"soup_pct": 0.0, "soup_judged": True, "admission_tested": False})

    def test_the_minimum_row_count_is_enough_for_the_admission_test(self):
        rows = rows_by_position(100, 50)
        problems, metrics = check_run(rows, texts_for(rows))
        self.assertEqual(len(problems), 1)
        self.assertIn("admission order (finished)", problems[0])
        self.assertTrue(metrics["admission_tested"])

    def test_admission_is_not_tested_when_no_outcome_is_on_record(self):
        # 500 rows that carry an id only: there is no outcome to compare between the halves, and the record says so
        rows = [{"id": f"m{i}"} for i in range(500)]
        problems, metrics = check_run(rows, texts_for(rows))
        self.assertEqual(problems, [])
        self.assertEqual(metrics, {"soup_pct": 0.0, "soup_judged": True, "admission_tested": False})

    def test_soup_bar_none_reports_the_rate_without_judging_it(self):
        rows = full_run()
        texts = texts_for(rows, lambda i: i % 10 < 3)
        problems, metrics = check_run(rows, texts, ok_key="ok_rescored", soup_bar=None)
        self.assertEqual(problems, [])
        self.assertEqual(metrics["soup_pct"], 30.0)
        self.assertFalse(metrics["soup_judged"])

    def test_closed_is_tested_when_present(self):
        # no "finish" key, so the finished outcome is not tested at all; thinking closed is, and here it
        # depends on position
        rows = [{"id": f"m{i}", "closed": i < 250} for i in range(500)]
        problems, _ = check_run(rows, texts_for(rows))
        self.assertEqual(len(problems), 1)
        self.assertIn("admission order (closed)", problems[0])

    def test_closed_is_not_tested_when_absent(self):
        rows = [{"id": f"m{i}", "finish": "stop"} for i in range(500)]
        _, metrics = check_run(rows, texts_for(rows))
        self.assertNotIn("admission_closed", metrics)

    def test_metrics_report_every_outcome_tested(self):
        rows = full_run(lambda i: i < 250)
        _, metrics = check_run(rows, texts_for(rows), ok_key="ok_rescored")
        self.assertEqual(metrics["admission_finished"], [100.0, 100.0, 1.0])
        self.assertEqual(metrics["admission_closed"], [100.0, 100.0, 1.0])
        first, second, p = metrics["admission_correct"]
        self.assertEqual((first, second), (100.0, 0.0))
        self.assertLess(p, 0.001)

    # ---- what decides each verdict ----
    def test_a_soup_rate_at_the_bar_passes(self):
        rows = full_run()
        problems, metrics = check_run(rows, texts_for(rows, lambda i: i % 5 == 0), ok_key="ok_rescored")   # 20.0% of 500
        self.assertEqual(problems, [])
        self.assertEqual(metrics["soup_pct"], 20.0)

    def test_a_soup_rate_just_above_the_bar_fails_with_that_one_problem(self):
        rows = full_run()
        texts = texts_for(rows, lambda i: i % 5 == 0 or i == 1)   # 101 of 500 = 20.2%: one generation past the bar
        problems, _ = check_run(rows, texts, ok_key="ok_rescored")
        self.assertEqual(problems, ["soup: 20.2% of generations end in non-Latin script (bar 20.0%)"])

    def test_the_soup_bar_argument_is_the_bar_that_decides(self):
        rows = full_run()
        texts = texts_for(rows, lambda i: i % 10 < 3)   # 30.0% of 500
        self.assertEqual(check_run(rows, texts, ok_key="ok_rescored", soup_bar=30.0)[0], [])
        problems, _ = check_run(rows, texts, ok_key="ok_rescored", soup_bar=29.9)
        self.assertEqual(problems, ["soup: 30.0% of generations end in non-Latin script (bar 29.9%)"])

    def test_soup_bar_none_adds_no_soup_problem_to_a_run_that_fails_the_admission_check(self):
        rows = full_run(lambda i: i < 250)
        texts = texts_for(rows, lambda i: i % 10 < 3)   # 30% of 500
        problems, metrics = check_run(rows, texts, ok_key="ok_rescored", soup_bar=None)
        self.assertEqual(len(problems), 1)
        self.assertIn("admission order (correct)", problems[0])
        self.assertEqual(metrics["soup_pct"], 30.0)
        self.assertFalse(metrics["soup_judged"])

    def test_finishing_that_depends_on_batch_position_fails_the_admission_check(self):
        rows = rows_by_position(500, 250)   # the first half finished, the second half hit the length cap
        problems, metrics = check_run(rows, texts_for(rows))
        self.assertEqual(len(problems), 1)
        self.assertIn("admission order (finished)", problems[0])
        self.assertEqual(metrics["admission_finished"][:2], [100.0, 0.0])
        self.assertLess(metrics["admission_finished"][2], 0.001)

    def test_a_row_without_a_finish_field_counts_as_not_finished(self):
        # the first half recorded why its text ended; the second half is an older-style row with no "finish"
        rows = [{"id": f"m{i}", "finish": "stop"} if i < 250 else {"id": f"m{i}"} for i in range(500)]
        problems, metrics = check_run(rows, texts_for(rows))
        self.assertTrue(mentioning(problems, "admission order (finished)"), problems)
        self.assertEqual(metrics["admission_finished"][:2], [100.0, 0.0])

    def test_finished_is_not_tested_when_no_row_carries_a_finish_field(self):
        rows = [{k: v for k, v in r.items() if k != "finish"} for r in full_run()]
        _, metrics = check_run(rows, texts_for(rows), ok_key="ok_rescored")
        self.assertEqual(sorted(metrics), ["admission_closed", "admission_correct", "admission_tested",
                                           "soup_judged", "soup_pct"])

    def test_an_evenly_spread_mixed_outcome_passes_the_admission_check(self):
        # every outcome is mixed (not all-true as in full_run) and spread evenly over the two halves
        rows = [{"id": f"m{i}", "finish": "stop" if i % 2 == 0 else "length", "closed": i % 3 == 0, "ok": i % 2 == 1}
                for i in range(500)]
        problems, metrics = check_run(rows, texts_for(rows), ok_key="ok")
        self.assertEqual(problems, [])
        self.assertEqual(metrics["admission_finished"], [50.0, 50.0, 1.0])
        self.assertEqual(metrics["admission_closed"][:2], [33.6, 33.2])   # 84 of 250 against 83 of 250
        self.assertEqual(metrics["admission_correct"], [50.0, 50.0, 1.0])

    def test_a_run_failing_both_checks_reports_both_problems(self):
        rows = full_run(lambda i: i < 250)
        texts = texts_for(rows, lambda i: i % 10 < 3)   # 30% of 500
        problems, _ = check_run(rows, texts, ok_key="ok_rescored")
        self.assertEqual(len(problems), 2)
        self.assertIn("soup", problems[0])
        self.assertIn("admission order (correct)", problems[1])

    def test_too_few_rows_still_judges_the_soup_rate(self):
        # 99 rows: below MIN_ROWS, so the admission test does not run and reports nothing (the metrics say so);
        # the soup check is all that is left, and it still decides
        rows = rows_by_position(99, 50)
        problems, metrics = check_run(rows, texts_for(rows, lambda i: i % 10 < 3))   # 30 of 99
        self.assertEqual(problems, ["soup: 30.3% of generations end in non-Latin script (bar 20.0%)"])
        self.assertEqual(metrics, {"soup_pct": 30.3, "soup_judged": True, "admission_tested": False})

    def test_a_row_with_no_generation_is_the_one_problem_and_nothing_else_is_tested(self):
        # an incomplete run is never called valid: the missing generation is the only problem, and the
        # tests that need every generation (fragment tails, outcomes by position) do not run
        rows = full_run()
        texts = texts_for(rows)
        del texts["m499"]
        problems, metrics = check_run(rows, texts, ok_key="ok_rescored")
        self.assertEqual(problems, ["generations: 1 of 500 rows have no saved generation"])
        self.assertEqual(metrics, {"soup_pct": None, "soup_judged": True, "admission_tested": False})

    def test_a_run_with_no_rows_is_a_problem(self):
        problems, metrics = check_run([], {})
        self.assertEqual(problems, ["rows: the run has no rows"])
        self.assertEqual(metrics, {"soup_pct": None, "soup_judged": True, "admission_tested": False})

    def test_ok_key_ok_rescored_judges_that_column_and_not_ok(self):
        rows = [dict(r, ok=i < 250) for i, r in enumerate(full_run())]   # `ok` depends on position, `ok_rescored` does not
        problems, metrics = check_run(rows, texts_for(rows), ok_key="ok_rescored")
        self.assertEqual(problems, [])
        self.assertEqual(metrics["admission_correct"], [50.0, 50.0, 1.0])

    def test_ok_key_ok_judges_that_column_and_not_ok_rescored(self):
        rows = [dict(r, ok=i < 250) for i, r in enumerate(full_run())]   # `ok` depends on position, `ok_rescored` does not
        problems, metrics = check_run(rows, texts_for(rows), ok_key="ok")
        self.assertEqual(len(problems), 1)
        self.assertIn("admission order (correct)", problems[0])
        self.assertEqual(metrics["admission_correct"][:2], [100.0, 0.0])
        self.assertLess(metrics["admission_correct"][2], 0.001)


class ScratchDirectoryCase(unittest.TestCase):
    """A test with its own empty directory, self.dir, removed afterwards"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def names_in_dir(self):
        """Every file and directory under self.dir, as sorted relative paths."""
        return sorted(p.relative_to(self.dir).as_posix() for p in self.dir.rglob("*"))


class SaveRunTest(ScratchDirectoryCase):
    """save_run(): a finished run and its generations, written so that neither is lost or replaced"""

    def save(self, result=None, compress=False):
        """save_run() of EXAMPLE_RESULT and EXAMPLE_TEXTS as run.json and run_texts.json[.gz] in self.dir;
        a `result` given is saved instead of the example result."""
        texts_path = self.dir / ("run_texts.json.gz" if compress else "run_texts.json")
        save_run(self.dir / "run.json", EXAMPLE_RESULT if result is None else result, texts_path, EXAMPLE_TEXTS,
                 compress=compress)

    def test_save_run_leaves_exactly_the_result_and_generations_files(self):
        self.save()
        self.assertEqual(self.names_in_dir(), ["run.json", "run_texts.json"])   # no *.tmp file is left behind

    def test_save_run_generations_file_reads_back_as_the_texts_given(self):
        self.save()
        self.assertEqual(json.loads((self.dir / "run_texts.json").read_text(encoding="utf-8")), EXAMPLE_TEXTS)

    def test_save_run_result_file_reads_back_as_the_result_given(self):
        self.save()
        self.assertEqual(json.loads((self.dir / "run.json").read_text(encoding="utf-8")), EXAMPLE_RESULT)

    def assert_file_is(self, name, expected):
        """The file `name` in self.dir holds exactly the text `expected`, and nothing but ASCII."""
        raw = (self.dir / name).read_bytes()
        self.assertTrue(raw.isascii(), f"{name} holds a non-ASCII character: {raw!r}")
        self.assertEqual(raw.decode("ascii"), expected)

    def test_save_run_writes_the_generations_as_ascii_escaped_json_indented_by_one_space(self):
        self.save()
        self.assert_file_is("run_texts.json", '{\n "m0": "The answer is 7.",\n "m1": "' + "\\u4e2d" * 30 + '"\n}')

    def test_save_run_writes_the_result_as_ascii_escaped_json_indented_by_one_space(self):
        self.save()
        self.assert_file_is("run.json",
                            '{\n'
                            ' "summary": {\n'
                            '  "name": "caf\\u00e9 run",\n'
                            '  "accuracy": 50.0\n'
                            ' },\n'
                            ' "rows": [\n'
                            '  {\n'
                            '   "id": "m0",\n'
                            '   "ok": true\n'
                            '  }\n'
                            ' ]\n'
                            '}')

    def test_save_run_creates_missing_parent_directories(self):
        save_run(self.dir / "new" / "deeper" / "run.json", EXAMPLE_RESULT,
                 self.dir / "generations" / "run_texts.json", EXAMPLE_TEXTS)
        self.assertEqual(self.names_in_dir(), ["generations", "generations/run_texts.json", "new", "new/deeper",
                                               "new/deeper/run.json"])

    def test_save_run_compress_writes_a_gzip_file_holding_the_texts(self):
        self.save(compress=True)
        self.assertEqual(self.names_in_dir(), ["run.json", "run_texts.json.gz"])
        with gzip.open(self.dir / "run_texts.json.gz", "rt", encoding="utf-8") as f:
            self.assertEqual(json.load(f), EXAMPLE_TEXTS)

    def test_save_run_writes_the_generations_before_the_result(self):
        # the result step is made to fail (json cannot encode this "result"): by then the generations file is
        # complete and in place, and neither a result file nor a temporary one was left
        with self.assertRaises(TypeError):
            self.save(result=object())
        self.assertEqual(self.names_in_dir(), ["run_texts.json"])
        self.assertEqual((self.dir / "run_texts.json").read_bytes(),
                         json.dumps(EXAMPLE_TEXTS, indent=1).encode("utf-8"))

    def test_save_run_writes_nothing_when_the_result_parent_is_a_file(self):
        blocker = self.dir / "blocker"   # stands where the result's directory should be
        blocker.write_bytes(b"not a directory")
        with self.assertRaises(FileExistsError):
            save_run(blocker / "run.json", EXAMPLE_RESULT, self.dir / "run_texts.json", EXAMPLE_TEXTS)
        self.assertEqual(self.names_in_dir(), ["blocker"])
        self.assertEqual(blocker.read_bytes(), b"not a directory")


class RefuseExistingTest(ScratchDirectoryCase):
    """save_run() and refuse_existing(): a file that already exists is refused, never replaced"""

    def test_save_run_refuses_when_the_result_file_exists(self):
        result_path = self.dir / "run.json"
        result_path.write_bytes(b"an older run")
        with self.assertRaises(SystemExit) as raised:
            save_run(result_path, EXAMPLE_RESULT, self.dir / "run_texts.json", EXAMPLE_TEXTS)
        self.assertIn(str(result_path), str(raised.exception))
        self.assertEqual(result_path.read_bytes(), b"an older run")
        self.assertEqual(self.names_in_dir(), ["run.json"])   # no generations file was written either

    def test_save_run_refuses_when_the_generations_file_exists(self):
        texts_path = self.dir / "run_texts.json"
        texts_path.write_bytes(b"older generations")
        with self.assertRaises(SystemExit) as raised:
            save_run(self.dir / "run.json", EXAMPLE_RESULT, texts_path, EXAMPLE_TEXTS)
        self.assertIn(str(texts_path), str(raised.exception))
        self.assertEqual(texts_path.read_bytes(), b"older generations")
        self.assertEqual(self.names_in_dir(), ["run_texts.json"])   # no result file was written either

    def test_refuse_existing_passes_while_no_path_exists(self):
        refuse_existing()
        refuse_existing(self.dir / "absent.json", self.dir / "absent_texts.json")

    def test_refuse_existing_names_the_one_path_that_exists(self):
        existing = self.dir / "exists.json"
        existing.write_bytes(b"")
        with self.assertRaises(SystemExit) as raised:
            refuse_existing(self.dir / "absent.json", existing)
        self.assertEqual(str(raised.exception), f"refusing to overwrite {existing}; choose a new name (ISSUES #20)")


class MainTest(ScratchDirectoryCase):
    """main(): the command line over a saved run; exit status 2 when a check fails"""

    def recheck(self, rows, soupy=lambda i: False, *flags):
        """main() on a saved run with these rows and generations, given `flags`: (exit status, the printed report)"""
        run_path = self.dir / "run.json"
        run_path.write_text(json.dumps(dict(rows=rows)), encoding="utf-8")
        self.dir.joinpath("run_texts.json").write_text(json.dumps(texts_for(rows, soupy)), encoding="utf-8")
        printed = io.StringIO()
        with mock.patch.object(sys, "argv", ["runcheck.py", str(run_path), *flags]), contextlib.redirect_stdout(printed):
            with self.assertRaises(SystemExit) as raised:
                main()
        return raised.exception.code, json.loads(printed.getvalue())

    def test_main_exits_2_when_the_soup_check_fails(self):
        code, report = self.recheck(full_run(), lambda i: i % 10 < 3)   # 30% of 500
        self.assertEqual(code, 2)
        self.assertEqual(report["problems"], ["soup: 30.0% of generations end in non-Latin script (bar 20.0%)"])

    def test_main_exits_0_for_a_clean_run(self):
        code, report = self.recheck(full_run())
        self.assertEqual(code, 0)
        self.assertEqual(report, {"run": "run", "soup_pct": 0.0, "soup_judged": True, "admission_tested": True,
                                  "admission_finished": [100.0, 100.0, 1.0], "admission_closed": [100.0, 100.0, 1.0],
                                  "admission_correct": [50.0, 50.0, 1.0], "problems": []})

    def test_main_soup_report_only_does_not_exit_2_for_a_soup_only_failure(self):
        code, report = self.recheck(full_run(), lambda i: i % 10 < 3, "--soup-report-only")
        self.assertEqual(code, 0)
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["soup_pct"], 30.0)   # the rate is still reported, only unjudged
        self.assertFalse(report["soup_judged"])

    def test_main_soup_report_only_still_exits_2_for_an_admission_failure(self):
        code, report = self.recheck(full_run(lambda i: i < 250), lambda i: False, "--soup-report-only")
        self.assertEqual(code, 2)
        self.assertEqual(len(report["problems"]), 1)
        self.assertIn("admission order (correct)", report["problems"][0])

    def test_main_reports_a_run_with_no_rows_and_exits_2(self):
        run_path = self.dir / "empty.json"
        run_path.write_text(json.dumps(dict(rows=[])), encoding="utf-8")
        printed = io.StringIO()
        with (mock.patch.object(sys, "argv", ["runcheck.py", str(run_path)]),
              contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO())):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual((raised.exception.code, json.loads(printed.getvalue())),
                         (2, {"run": "empty", "soup_pct": None, "soup_judged": True, "admission_tested": False,
                              "problems": ["rows: the run has no rows"]}))


class RealRunTest(unittest.TestCase):
    """Integration on the committed phase-7 `stack` finals; each test is skipped when a run's files are absent"""

    def load_run(self, run):
        """(rows with the rescored verdict merged in by id, generations) for a committed run"""
        run_path = HERE / f"{run}.json"
        texts_path, rescored_path = HERE / f"{run}_texts.json.gz", HERE / f"{run}_rescored.json"
        for p in (run_path, texts_path, rescored_path):
            if not p.exists():
                raise unittest.SkipTest(f"{p.name} is not in this checkout")
        with gzip.open(texts_path, "rt", encoding="utf-8") as f:
            texts = json.load(f)
        rescored = json.loads(rescored_path.read_text(encoding="utf-8"))["rows"]
        ok = {r["id"]: r["ok_rescored"] for r in rescored}
        rows = json.loads(run_path.read_text(encoding="utf-8"))["rows"]
        for r in rows:   # the merge main() performs when the verdicts live in <run>_rescored.json
            r["ok_rescored"] = ok[r["id"]]
        return rows, texts

    def test_stack_full_final_is_caught(self):
        rows, texts = self.load_run("p7_stack_full")
        problems, _ = check_run(rows, texts, ok_key="ok_rescored")
        self.assertTrue(mentioning(problems, "soup"), problems)
        self.assertTrue(mentioning(problems, "admission order (correct)"), problems)

    def test_stack_half_final_passes(self):
        rows, texts = self.load_run("p7_stack_half")
        problems, _ = check_run(rows, texts, ok_key="ok_rescored")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
