"""Regression tests for phase3c/rescore.py: answer extraction, including the
ISSUES #19 fixes (applied 2026-11-23 and 2026-11-24; see the rescore module docstring).
  a. a $math$ span that ENDS the payload is read before any '$' is stripped
     (the trailing period is dropped first, "$$...$$" counts, and an escaped
     \\$ inside the span is content);
  a2. a span followed by a linking word and a plain value states that value, and
     spans are searched before any '$' is stripped;
  d. an escaped dollar is a dollar sign (phase3b/scorer.py, canonical);
  b. a $$...$$ payload is read as its math content;
  c. a stray </think> token inside the answer section is removed first.
The preserved extraction order and cleanup are pinned as well; two real
committed generations are scored end to end (skipped when the data is absent),
and answer lines copied verbatim from committed generations are checked without
the data files (RealAnswerLineTest).

Standard library only. Usage (from the repository root):
  python -B -m unittest -v phase3c/test_rescore.py
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
sys.path.insert(0, str(HERE))  # the way rescore.py itself reaches phase3b
import rescore  # noqa: E402
from rescore import answers_match, canonical, clean, extract_answer  # noqa: E402

RUN = "p3c_c3_original"  # <RUN>_texts.json.gz holds the generations, <RUN>.json the gold


class CleanTest(unittest.TestCase):
    """clean(): the text after an answer marker -> the answer string"""

    # ---- fixed 2026-11-23 ----
    def test_math_span_survives_the_trailing_period(self):
        # defect a: the period is dropped first, so the span's closing '$' is the last character
        self.assertEqual(clean('The eighth term of the sequence is $\\frac{243}{625}$.'), '\\frac{243}{625}')

    def test_math_span_is_canonicalised(self):
        # defect a, the case named in the module docstring (was scored as "7")
        self.assertEqual(clean('The resulting complex number is $-2 + 7i$.'), '-2+7i')

    def test_math_span_ending_the_payload_wins_over_an_earlier_span(self):
        # the answer is the span that ENDS the payload (defect a); the earlier $OAB$ is not it
        self.assertEqual(clean('The area of triangle $OAB$ is $3\\sqrt{5}$.'), '3\\sqrt{5}')

    def test_double_dollar_payload_is_read_as_math(self):
        # defect b: was scored as "5", the last number of the prose remainder
        self.assertEqual(clean('$$5r^5$$'), '5r^5')

    # ---- preserved ----
    def test_bare_math_span(self):
        self.assertEqual(clean('$72$'), '72')

    def test_split_at_the_last_equals_sign(self):
        # intentional: a gold such as "x=5" is still compared with "5"
        self.assertEqual(clean('**Answer:** $x = 5$'), '5')

    def test_prose_remainder_gives_the_last_number(self):
        self.assertEqual(clean('the answer is approximately 3.14159'), '3.14159')


class ExtractAnswerTest(unittest.TestCase):
    """extract_answer(): a generation -> its answer, None when thinking never closed"""

    # ---- fixed 2026-11-23 ----
    def test_marker_line_with_a_math_span_payload(self):
        self.assertEqual(extract_answer('</think>\nThe final answer is $\\frac{1}{2}$.'), '\\frac{1}{2}')

    def test_math_span_ending_the_answer_line_is_the_answer(self):
        # defect a: read once the trailing period is dropped, before any '$' is stripped (was "7")
        self.assertEqual(extract_answer('</think>\nThe final answer is $-2 + 7i$.'), canonical('-2 + 7i'))

    def test_display_math_span_ending_the_payload_is_the_answer(self):
        # "$$...$$" ends the payload too: the extra closing dollar belongs to the span (was "5")
        self.assertEqual(extract_answer('</think>\n$$5r^5$$'), canonical('5r^5'))

    def test_money_span_ending_the_answer_line_is_the_answer(self):
        # an escaped \$ is content, not a delimiter: the span reads as \$18.90. The expectation is
        # canonical() of that content, so it follows the scorer's treatment of the escaped dollar.
        self.assertEqual(extract_answer('</think>\nThe answer is $\\$18.90$.'), canonical('\\$18.90'))

    def test_math_span_inside_the_answer_line_is_not_the_answer(self):
        # a2: the span does not end the payload and the sentence states its value
        self.assertEqual(extract_answer('</think>\nThe value of $x$ is equal to 4.'), '4')

    def test_stray_think_token_with_payload_on_the_next_line(self):
        # defect c: the token was read as the payload
        self.assertEqual(extract_answer('we reason here</think>\n\n**Answer:** </think>\n42'), '42')

    def test_stray_think_token_with_payload_on_the_same_line(self):
        self.assertEqual(extract_answer('we reason here</think>\n\n**Answer:** </think> 42'), '42')

    def test_stray_think_token_with_a_math_span_on_the_same_line(self):
        # the case above passed before the fix too: the prose fallback happens to take "42"
        self.assertEqual(extract_answer('we reason here</think>\n\n**Answer:** </think> $\\frac{1}{2}$'), '\\frac{1}{2}')

    def test_stray_think_token_after_the_answer_is_marker(self):
        # with the marker spelled out in words rather than a bolded "Answer:" label. Like the "42"
        # case above, this passed before the fix too: the prose fallback happens to take "5"
        self.assertEqual(extract_answer('we reason here</think>\nThe answer is </think> 5'), '5')

    def test_stray_think_token_after_the_payload(self):
        # the token is dropped wherever it sits on the line. A LaTeX payload is trusted whole, so
        # a trailing token would be read into the answer
        self.assertEqual(extract_answer('we reason here</think>\nThe answer is \\frac{1}{2} </think>'), '\\frac{1}{2}')

    # ---- preserved ----
    def test_boxed_wins_over_a_later_marker(self):
        self.assertEqual(extract_answer('thinking \\boxed{7} more</think>\nAnswer: 8'), '7')

    def test_last_boxed_wins(self):
        self.assertEqual(extract_answer('thinking \\boxed{3} more \\boxed{7}</think>\nAnswer: 8'), '7')

    def test_truncated_thinking_is_no_answer(self):
        self.assertIsNone(extract_answer('still thinking, never closed'))

    def test_no_think_does_not_require_a_closed_think_block(self):
        self.assertIsNotNone(extract_answer('still thinking, never closed', no_think=True))

    def test_marker_payload_on_the_next_line(self):
        self.assertEqual(extract_answer('x</think>\n**Final Answer:**\n\\frac{1}{2}'), '\\frac{1}{2}')

    def test_no_marker_falls_back_to_the_sections_last_line(self):
        self.assertEqual(extract_answer('x</think>\nSo the area is 12.'), '12')

    def test_fallback_takes_the_last_of_several_lines(self):
        self.assertEqual(extract_answer('x</think>\nThe base is 4.\nSo the area is 12.'), '12')


class AnswersMatchTest(unittest.TestCase):
    """answers_match() from phase3b/scorer.py, as the rescorer uses it"""

    def test_spaces_are_ignored(self):
        self.assertTrue(answers_match('-2 + 7i', '-2+7i'))


class RealGenerationTest(unittest.TestCase):
    """Integration on committed generations; each test is skipped when the run's files are absent"""

    def setUp(self):
        texts_path, run_path = HERE / f"{RUN}_texts.json.gz", HERE / f"{RUN}.json"
        for p in (texts_path, run_path):
            if not p.exists():
                raise unittest.SkipTest(f"{p.name} is not in this checkout")
        with gzip.open(texts_path, "rt", encoding="utf-8") as f:
            self.texts = json.load(f)
        self.gold = {r["id"]: r["gold"] for r in json.loads(run_path.read_text(encoding="utf-8"))["rows"]}

    def assert_scores_correct(self, pid):
        pred = extract_answer(self.texts[pid])
        self.assertTrue(answers_match(self.gold[pid], pred), f"{pid}: extracted {pred!r}, gold is {self.gold[pid]!r}")

    def test_m28_bold_math_span_on_the_sections_last_line(self):
        # no marker line; the answer sits in **$-2 + 7i$** (defect a: was scored as "7")
        self.assert_scores_correct("m28")

    def test_m20_double_dollar_payload_after_the_marker(self):
        # "**Final Answer:**", then $$ 6 + 9i $$ on the next line (defect b: was scored as "9")
        self.assert_scores_correct("m20")


class RealAnswerLineTest(unittest.TestCase):
    """Answer lines copied verbatim from committed generations and trimmed to the line that
    decides the answer (run and problem id in each comment), so no data file is needed. Each must
    extract its gold answer. These are the lines on which a naive "take the last $...$ span
    anywhere" rule was rejected: it read "$\\$18.90$" as a lone "\\" and "$x$ is equal to 4" as "x".
    """

    def assert_extracts_gold(self, text, gold):
        pred = extract_answer(text)
        self.assertTrue(answers_match(gold, pred), f"extracted {pred!r}, gold is {gold!r}")

    # ---- money: an escaped \$ inside the span is content ----
    def test_money_span_on_the_marker_line(self):
        # p7_uperp_on_loop m459
        self.assert_extracts_gold('we reason here</think>\n**Answer:** $\\$18.90$', '\\$18.90')

    def test_money_span_on_the_sections_last_line(self):
        # p9_interleaved m488: no marker line
        self.assert_extracts_gold(
            'we reason here</think>\nTherefore, the original price of the shirt was $\\$36$.', '\\$36')

    def test_money_answer_with_a_bare_number_gold(self):
        # d: the span reads as \$10 and an escaped dollar is a dollar sign, so it matches "10"
        # (p10_fullbudget m270: "how many more dollars does Pam pay than Susan?")
        self.assert_extracts_gold('we reason here</think>\nPam pays $\\$10$ more than Susan.', '10')

    # ---- a $math$ span that does not end the payload ----
    def test_is_equal_to_line_gives_the_number(self):
        # p10_fullbudget m262: the span does not end the payload and the sentence states its value (a2)
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$x$ is equal to **4**.', '4')

    def test_expression_span_before_trailing_prose_is_the_answer(self):
        # p10_fullbudget m171: the span does not end the payload, no value is stated, and it IS the
        # answer. Any change to the stated-value rule must keep this one green.
        self.assert_extracts_gold(
            'we reason here</think>\n**Answer:**\nLouis and Jack eat altogether **$\\frac{13}{15}$** of the pie.',
            '\\frac{13}{15}')

    def test_value_of_an_expression_line_gives_the_number(self):
        # a2, p3c_c3_original m42: "Find the value of $a + b$"
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\nThe value of $a + b$ is 4.', '4')

    def test_value_of_a_variable_line_gives_the_number(self):
        # a2, p10_fullbudget m56: "What is the value of $N$?"; no marker line
        self.assert_extracts_gold('we reason here</think>\nThe value of $N$ is 11.', '11')

    def test_a_span_followed_by_a_unit_stays_the_answer(self):
        # a2: "is $\sqrt{5}$ meters" states no value after the span (p10_fullbudget m85)
        self.assert_extracts_gold('we reason here</think>\nThe radius of the tank is $\\sqrt{5}$ meters.',
                                  '\\sqrt{5}')

    def test_a_span_followed_by_a_linking_phrase_and_prose_stays_the_answer(self):
        # a2: only a plain number, fraction or \frac after the phrase counts as a stated value
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\nThe value of $a + b$ is what was asked.',
                                  'a + b')

    def test_a_number_after_the_linking_phrase_that_is_not_alone_is_not_taken(self):
        # a2: "is 4 dollars" is not a stated value; the span stays the answer
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\nThe value of $x$ is 4 dollars.', 'x')

    def test_a_stated_value_may_be_bold_or_wrapped_in_dollars(self):
        self.assertEqual(extract_answer('we reason here</think>\n**Answer:**\nThe value of $x$ is **-128**.'),
                         '-128')
        self.assertEqual(extract_answer('we reason here</think>\n**Answer:**\nThe value of $x$ is $\\frac{1}{2}$.'),
                         '\\frac{1}{2}')

    def test_a_stated_value_with_thousands_separator(self):
        # p3c_c3_original m336: "the value of $513^2 - 487^2$ is 26,000", gold 26000
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\nThe value of $513^2 - 487^2$ is 26,000.',
                                  '26000')

    def test_a_few_words_may_stand_between_the_span_and_is(self):
        # a2, three runs on p-m430: "$\frac{9}{2}$ expressed as a decimal is **4.5**"
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$\\frac{9}{2}$ expressed as a decimal is **4.5**.',
                                  '4.5')

    def test_a_span_that_opens_the_payload_survives_a_trailing_unit(self):
        # p7_stack_half m220: the dollars are not stripped before the spans are searched
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$\\frac{16}{27}$ gallons.', '\\frac{16}{27}')
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$\\sqrt{53}$ is the simplest radical form.',
                                  '\\sqrt{53}')

    def test_an_ordinal_span_is_its_number(self):
        # p11 a0_full_final m379: no marker line; the last line names the grade in bold math
        self.assert_extracts_gold(
            'we reason here</think>\nThe number of male bus riders is closest to $135\\%$ of the number of '
            'female bus riders in the **$12^{\\text{th}}$ grade**.', '12')

    # ---- which line is read, and how far the stated-value rule reaches ----
    def test_the_last_marker_line_wins_in_either_order(self):
        # two lines carry an answer marker: the later one is the model's last word
        self.assert_extracts_gold('we reason here</think>\nThe answer is $4$.\nFinal answer: 7.', '7')
        self.assert_extracts_gold('we reason here</think>\nFinal answer: 7.\nThe answer is $4$.', '4')

    def test_more_than_four_words_before_the_linking_word_is_not_a_stated_value(self):
        # a2 allows at most four words between the span and the linking word; five leave the span as the answer
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$x$ expressed as a decimal here is 4.', 'x')

    def test_a_text_command_inside_the_span_gives_its_content(self):
        self.assert_extracts_gold('we reason here</think>\n**Answer:**\n$\\text{even}$', 'even')


class RescoreRunTest(unittest.TestCase):
    """rescore(): a whole run file re-scored from the generations beside it"""

    def test_a_run_that_records_no_cap_hit_rate_is_re_scored_with_none(self):
        # runs made by the phase-8 runner carry no cap-hit rate in their summary
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = dict(summary=dict(name="x", mean_tokens=12), rows=[dict(id="m0", gold="7")])
            (tmp / "x.json").write_text(json.dumps(run), encoding="utf-8")
            (tmp / "x_texts.json").write_text(json.dumps({"m0": "we think</think>\nThe answer is $7$."}), encoding="utf-8")
            with mock.patch.object(rescore, "__file__", str(tmp / "rescore.py")):   # rescore() works beside itself
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed):
                    problems = rescore.rescore("x")
            out = json.loads((tmp / "x_rescored.json").read_text(encoding="utf-8"))
        self.assertEqual(problems, [])
        self.assertEqual((out["summary"]["accuracy"], out["summary"]["cap_hit_rate"], out["rows"][0]["ok_rescored"]),
                         (100.0, None, True))
        self.assertEqual(out["summary"]["rescored_with"], rescore.RESCORED_WITH)


if __name__ == "__main__":
    unittest.main()
