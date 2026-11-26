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

Third rule, 2026-11-26 (ISSUES #30 and #31; see the rescore module docstring).
  b3. A box is read to its matching brace at any depth (LastBoxedTest).
  a3. An answer without a box is right when the last line carrying an answer
      marker is right or, else, the section's last line is right: which lines
      are the section (AnswerLinesTest), the marked line (MarkedLineTest),
      whether a line can be a conclusion (ConcludesTest, LastLineAnswerTest),
      and the verdict with the rule that decided it (JudgeTest). The cleanup
      this needed is in CleanTest, under "third rule".
  Reviewed 2026-11-26, after two code reviews (the tests of it are marked
  "reviewed"): a blank box (empty, or holding only spacing) is passed over, the
  text is read in one pass and of two boxes one inside the other the inner one
  is the answer (LastBoxedTest); a bold label in front of a value goes with it,
  and a long run of closing brackets is removed in one pass (CleanTest); a line
  that stops at an operator, or is a heading or a list number, is not a
  conclusion (ConcludesTest); `rule` names the line the compared answer came
  from, right or wrong (JudgeTest); a re-score records the thinking flag and
  each row's rule (RescoreRunTest).

Standard library only. Usage (from the repository root):
  python -B -m unittest -v phase3c/test_rescore.py
"""

import contextlib
import gzip
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the way rescore.py itself reaches phase3b
import rescore  # noqa: E402
from rescore import (answers_match, answer_lines, canonical, clean, concludes, extract_answer, judge,  # noqa: E402
                     last_boxed, last_line_answer, marked_line)

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

    # ---- third rule, 2026-11-26 ----
    def test_a_bold_label_reads_with_its_value(self):
        # opening stars always go, and since the second review the label goes with them: before, the label
        # itself was read as the answer ("Result")
        self.assertEqual(clean('**Result:** 326.5'), '326.5')

    def test_the_opening_stars_of_a_bold_value_go_even_when_it_never_closes(self):
        # a generation cut off inside the bold: with its stars the value is not a number the matcher can read
        self.assertEqual(clean('**\\frac{1}{2}'), '\\frac{1}{2}')

    def test_a_bold_value_keeps_its_closing_stars_without_a_period(self):
        # the earlier rules stripped them, and the sentence read its math span ("x")
        self.assertEqual(clean('The value of $x$ is **4**'), '4')

    def test_a_bold_value_keeps_its_closing_stars_with_a_period(self):
        self.assertEqual(clean('The value of $x$ is **4**.'), '4')

    def test_a_period_then_stars_that_close_nothing_go_in_that_order(self):
        # once split at "=", the payload is " \frac{4}{3}$**."
        self.assertEqual(clean('**$y = \\frac{4}{3}$**.'), '\\frac{4}{3}')

    def test_stars_that_close_nothing_go_after_a_value(self):
        self.assertEqual(clean('4**.'), '4')

    def test_a_dollar_then_a_period_then_a_bracket_that_nothing_opened_go_in_that_order(self):
        self.assertEqual(clean('\\frac{3840}{289}$.)'), '\\frac{3840}{289}')

    def test_a_bracket_that_nothing_opened_goes(self):
        self.assertEqual(clean('2)'), '2')

    def test_the_latex_closing_parenthesis_is_not_a_bracket_to_remove(self):
        # "\)" is left for canonical(); removed here, it would leave a lone backslash
        self.assertEqual(clean('\\( k = 2 \\)'), '2')

    def test_a_list_keeps_its_brackets(self):
        self.assertEqual(clean('(3,5).'), '(3,5)')

    # ---- reviewed 2026-11-26: a bold label in front of the value; half-open intervals ----
    def test_a_bold_label_in_front_of_a_bold_value_reads_the_number(self):
        self.assertEqual(clean('**Answer:** **1,251**'), '1,251')

    def test_a_bold_label_in_front_of_a_bold_value_reads_the_choice_letter(self):
        # The brief was corrected on 2026-11-26: this reads "E", not "(E)". Once the label and the value's
        # closing stars are gone, "(E)" goes through the ordinary prose remainder, which takes the last word
        self.assertEqual(clean('**Answer:** **(E)**'), 'E')
        # what matters: the gold's parentheses and the reading are the same choice
        self.assertTrue(answers_match('\\text{(E)}', clean('**Answer:** **(E)**')))

    def test_a_bold_value_with_a_colon_inside_is_left_alone_by_the_label_rule(self):
        # The brief was corrected on 2026-11-26: this reads "5", not "x: 5". The label rule leaves the payload
        # alone (it needs the colon right before the closing stars); the ordinary prose remainder then takes
        # the last number, the reading it gives the payload without its stars
        self.assertEqual(clean('x: 5'), '5')
        self.assertEqual(clean('**x: 5**'), '5')
        # had the label rule applied, there would be nothing to strip to, and the payload would read None
        self.assertIsNotNone(clean('**x: 5**'))
        self.assertIsNone(clean('**Answer:**'))

    def test_a_half_open_interval_keeps_its_closing_bracket(self):
        self.assertEqual(clean('[2, 5)'), '[2,5)')

    def test_a_half_open_interval_after_a_variable_keeps_it_and_its_closing_bracket(self):
        # The brief was corrected on 2026-11-26: clean() splits at "=" only, never at "\in", so the variable
        # stays. The matcher then drops the leading variable on both sides
        self.assertEqual(clean('x \\in [0, 5)'), 'x\\in[0,5)')
        self.assertEqual(clean('x = [0, 5)'), '[0,5)')
        self.assertTrue(answers_match('x \\in [0, 5)', clean('x \\in [0, 5)')))
        self.assertFalse(answers_match('x \\in [0, 3)', clean('x \\in [0, 5)')))

    def test_a_union_of_intervals_keeps_the_brackets_it_needs(self):
        self.assertEqual(clean('[0,1) \\cup (2,3)'), '[0,1)\\cup(2,3)')

    def test_a_long_run_of_closing_brackets_is_removed_in_one_pass(self):
        started = time.perf_counter()
        self.assertEqual(clean('5' + ')' * 80000), '5')
        self.assertLess(time.perf_counter() - started, 1.0)


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

    def test_no_think_reads_a_concluding_line_without_a_closed_think_block(self):
        # Changed 2026-11-26 (the third rule, ISSUES #31). Until then the section's last line was read
        # whatever it said, and this test asserted that 'still thinking, never closed' gave an answer
        # under no_think=True. That line cannot be a conclusion (concludes() is False for a line that
        # stops mid-expression and has spaces), so a fragment is no longer read. The intent is the
        # same as before: with thinking off, the output needs no </think>.
        self.assertEqual(extract_answer('So the area is 12.', no_think=True), '12')
        self.assertIsNone(extract_answer('still thinking, never closed', no_think=True))

    def test_marker_payload_on_the_next_line(self):
        self.assertEqual(extract_answer('x</think>\n**Final Answer:**\n\\frac{1}{2}'), '\\frac{1}{2}')

    def test_no_marker_falls_back_to_the_sections_last_line(self):
        self.assertEqual(extract_answer('x</think>\nSo the area is 12.'), '12')

    def test_fallback_takes_the_last_of_several_lines(self):
        self.assertEqual(extract_answer('x</think>\nThe base is 4.\nSo the area is 12.'), '12')

    # ---- third rule, 2026-11-26 ----
    def test_a_box_nested_two_brace_levels_deep_is_the_answer(self):
        # read to its matching brace at any depth (ISSUES #30); the one-level pattern missed it
        self.assertEqual(extract_answer('x</think>\nThe final answer is $\\boxed{\\frac{\\sqrt{3}}{3}}$.'),
                         '\\frac{\\sqrt{3}}{3}')

    def test_the_marked_line_is_the_answer_even_when_a_later_line_concludes(self):
        # one string for a caller that wants an answer, not the verdict: the box, else the marked line,
        # else the last line. The verdict is judge(), which reads this generation's last line as right
        self.assertEqual(extract_answer('x</think>\n**Answer:**\n$x$ is 256\nThe value of $x$ is **4**.'), '256')

    def test_no_answer_for_a_last_line_that_stops_mid_expression(self):
        self.assertIsNone(extract_answer('x</think>\nSo $S = z + z^{-1'))


class LastBoxedTest(unittest.TestCase):
    """last_boxed(): the content of the last \\boxed{...} of a generation, read to its matching brace"""

    def test_reads_what_the_box_holds(self):
        self.assertEqual(last_boxed('The final answer is $\\boxed{42}$.'), '42')

    def test_reads_a_box_nested_two_brace_levels_deep(self):
        # the one-level pattern this replaces missed it (ISSUES #30), and the answer counted as not boxed
        self.assertEqual(last_boxed('The final answer is $\\boxed{\\frac{\\sqrt{3}}{3}}$.'), '\\frac{\\sqrt{3}}{3}')

    def test_reads_a_box_nested_deeper_still(self):
        self.assertEqual(last_boxed('The final answer is $\\boxed{\\frac{\\sqrt{\\sqrt{2}}}{2}}$.'),
                         '\\frac{\\sqrt{\\sqrt{2}}}{2}')

    def test_reads_the_last_box_of_several(self):
        self.assertEqual(last_boxed('$\\boxed{3}$ is too small, so the answer is $\\boxed{7}$'), '7')

    def test_reads_a_box_that_stands_inside_the_thinking(self):
        # looked for in the whole generation, whatever follows it
        self.assertEqual(last_boxed('thinking \\boxed{7} more</think>\nAnswer: 8'), '7')

    def test_returns_none_when_there_is_no_box(self):
        self.assertIsNone(last_boxed('The final answer is 42.'))

    def test_a_blank_box_is_not_an_answer(self):
        # Changed 2026-11-26 (second review): a blank box used to be read as "" and, being a box, to end the
        # search. It is passed over now: quoted from the instruction, it holds no answer
        self.assertIsNone(last_boxed('answer \\boxed{}'))

    def test_a_blank_box_after_a_real_one_is_passed_over(self):
        self.assertEqual(last_boxed('\\boxed{42} then \\boxed{}'), '42')
        self.assertEqual(last_boxed('\\boxed{42} then \\boxed{\\,}'), '42')

    def test_a_blank_box_before_a_real_one_is_passed_over(self):
        self.assertEqual(last_boxed('\\boxed{} then \\boxed{42}'), '42')

    def test_a_box_that_holds_only_spacing_is_not_an_answer(self):
        # like "\boxed{}", it quotes the instruction's empty box: LaTeX's spacing and blanks hold no answer
        for box in ('\\boxed{}', '\\boxed{ \\ }', '\\boxed{\\,}', '\\boxed{\\;}', '\\boxed{\\!}', '\\boxed{~}', '\\boxed{  }'):
            with self.subTest(box=box):
                self.assertIsNone(last_boxed('answer ' + box))

    def test_skips_an_unclosed_box(self):
        self.assertIsNone(last_boxed('The final answer is $\\boxed{42'))

    def test_an_unclosed_box_after_a_closed_one_leaves_that_one_the_last(self):
        self.assertEqual(last_boxed('$\\boxed{7}$, or is it $\\boxed{42'), '7')

    # ---- reviewed 2026-11-26: one pass over the text ----
    def test_a_box_nested_inside_an_unclosed_box_is_read(self):
        # the generation was cut off inside the outer box; the inner one is complete
        self.assertEqual(last_boxed('\\boxed{\\text{No. \\boxed{7}'), '7')

    # ---- of two boxes one inside the other, the inner one is the answer: it is the last to open ----
    def test_of_two_boxes_one_inside_the_other_the_inner_one_is_the_answer(self):
        self.assertEqual(last_boxed('the answer is \\boxed{\\text{see } \\boxed{5}}'), '5')

    def test_two_boxes_side_by_side_give_the_last(self):
        self.assertEqual(last_boxed('\\boxed{1} \\boxed{2}'), '2')

    def test_a_blank_outer_box_with_a_real_inner_box_gives_the_inner(self):
        self.assertEqual(last_boxed('\\boxed{ \\boxed{7}}'), '7')

    def test_a_closing_brace_that_no_box_opened_does_not_hide_the_box(self):
        self.assertEqual(last_boxed('} \\boxed{3}'), '3')

    def test_a_box_written_with_a_space_before_its_brace_is_not_read(self):
        # a known limit, the same under the rule this replaces
        self.assertIsNone(last_boxed('\\boxed {5}'))

    def test_a_generation_of_unclosed_boxes_is_passed_over_in_one_pass(self):
        text = '\\boxed{' * 16000
        self.assertIsNone(last_boxed(text))
        started = time.perf_counter()
        self.assertEqual(judge('7', text), (False, None, None))
        self.assertLess(time.perf_counter() - started, 1.0)


class AnswerLinesTest(unittest.TestCase):
    """answer_lines(): the non-empty lines of the answer section"""

    def test_gives_the_non_empty_lines_after_the_thinking(self):
        self.assertEqual(answer_lines('we reason here</think>\nThe answer is 4.\n\nSo **4**.\n'),
                         ['The answer is 4.', 'So **4**.'])

    def test_gives_nothing_of_the_thinking_itself(self):
        self.assertEqual(answer_lines('The area might be 7.</think>\nSo the area is 12.'), ['So the area is 12.'])

    def test_gives_the_whole_output_when_thinking_is_off(self):
        self.assertEqual(answer_lines('The answer is 4.\nSo **4**.', no_think=True), ['The answer is 4.', 'So **4**.'])

    def test_gives_no_lines_when_the_thinking_never_closes(self):
        self.assertEqual(answer_lines('still thinking, never closed'), [])

    def test_gives_no_lines_when_nothing_follows_the_thinking(self):
        for text in ('we reason here</think>', 'we reason here</think>\n  \n\t'):
            with self.subTest(text=text):
                self.assertEqual(answer_lines(text), [])

    def test_drops_a_stray_think_token_inside_the_section(self):
        # defect c: in stop-token-boosted arms the model emits </think> again on the answer line
        self.assertEqual(answer_lines('we reason here</think>\n**Answer:** </think>\n42'), ['**Answer:**', '42'])

    def test_strips_each_line(self):
        self.assertEqual(answer_lines('x</think>\n  The answer is 4.  \n'), ['The answer is 4.'])


class MarkedLineTest(unittest.TestCase):
    """marked_line(): the last line carrying an answer marker, sliced at the marker and read as an answer"""

    def test_reads_what_follows_the_marker(self):
        self.assertEqual(marked_line(['The answer is 4.']), '4')

    def test_reads_the_last_of_several_marked_lines(self):
        self.assertEqual(marked_line(['The answer is 4.', 'That seems off.', 'The value is 7.']), '7')

    def test_reads_the_next_line_when_the_marker_ends_the_line(self):
        self.assertEqual(marked_line(['**Answer:**', '\\frac{1}{2}']), '\\frac{1}{2}')

    def test_returns_none_when_no_line_carries_a_marker(self):
        self.assertIsNone(marked_line(['So the area is 12.', 'So the number of sides is 4.']))

    def test_returns_none_when_the_only_marker_ends_the_section(self):
        self.assertIsNone(marked_line(['**Final Answer:**']))

    def test_returns_none_for_an_empty_section(self):
        self.assertIsNone(marked_line([]))


class ConcludesTest(unittest.TestCase):
    """concludes(): whether a line can be a model's conclusion"""

    # ---- one bare value ----
    def test_one_bare_value_can_be_a_conclusion(self):
        for line in ('90', '(C)', '**1**'):
            with self.subTest(line=line):
                self.assertTrue(concludes(line))

    # ---- the line ends a sentence, or closes a bracket, a math span or a bold value ----
    def test_a_line_that_ends_a_sentence_can_be_a_conclusion(self):
        for line in ('There are **9** divisors.', 'So the answer is 9!', 'Is the answer 9?'):
            with self.subTest(line=line):
                self.assertTrue(concludes(line))

    def test_a_line_that_closes_a_bracket_can_be_a_conclusion(self):
        self.assertTrue(concludes('(or equivalently $(a+5)(b+2)$)'))

    def test_a_line_that_closes_a_math_span_can_be_a_conclusion(self):
        self.assertTrue(concludes('$$x = 5$$'))

    def test_a_line_that_closes_a_bold_value_can_be_a_conclusion(self):
        self.assertTrue(concludes('The value of $x$ is **4**'))

    # ---- a label followed by a value ----
    def test_a_bold_label_and_a_value_can_be_a_conclusion(self):
        for line in ('**Total ways:** 3', '**Result:** 326.5'):
            with self.subTest(line=line):
                self.assertTrue(concludes(line))

    def test_a_label_and_a_value_can_be_a_conclusion(self):
        self.assertTrue(concludes('Result: $a+b = 0$.'))

    # ---- a line that stops mid-expression and has spaces: not a conclusion ----
    def test_a_math_span_left_open_is_not_a_conclusion(self):
        for line in ('So $S = z + z^{-1', 'The denominator is $x^2 - 5x -'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_line_cut_off_after_an_equals_sign_is_not_a_conclusion(self):
        for line in ('$480^2 = 2', '$$f(2, 0) = f(1,'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_line_cut_off_mid_sentence_is_not_a_conclusion(self):
        self.assertFalse(concludes('Factors of -4 that sum to'))

    def test_a_step_heading_cut_off_is_not_a_conclusion(self):
        for line in ('**Step 2. Simplify the equation to', '**Step 10: Simplify the actually equation', '**Step 2',
                     '**Step 3m'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_remark_cut_off_after_a_colon_is_not_a_conclusion(self):
        self.assertFalse(concludes('Wait: f(1,'))

    # ---- reviewed 2026-11-26: a line that stops at an operator, or is a heading or a list number ----
    def test_a_line_that_stops_at_an_operator_is_not_a_conclusion(self):
        for line in ('= 1.5 \\cdot', 'x^2 +', '5 +', 'a /'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_line_that_stops_at_an_equals_sign_is_not_a_conclusion(self):
        self.assertFalse(concludes('value ='))

    def test_a_bare_list_of_arguments_cut_off_is_not_a_conclusion(self):
        self.assertFalse(concludes('f(1,'))

    def test_a_step_heading_that_ends_a_sentence_is_not_a_conclusion(self):
        for line in ('**Step 1.', 'Step 2.'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_list_number_is_not_a_conclusion(self):
        for line in ('3)', '**3)'):
            with self.subTest(line=line):
                self.assertFalse(concludes(line))

    def test_a_value_in_brackets_or_in_bold_is_still_a_conclusion(self):
        # unlike a list number ("3)"), these state a value
        for line in ('(3)', '**9**'):
            with self.subTest(line=line):
                self.assertTrue(concludes(line))

    def test_a_bare_number_with_a_period_is_still_a_bare_value(self):
        self.assertTrue(concludes('2.'))

    def test_a_value_followed_by_stars_that_close_nothing_is_still_a_conclusion(self):
        self.assertTrue(concludes('42**'))


class LastLineAnswerTest(unittest.TestCase):
    """last_line_answer(): the section's last line read as an answer, when it can be a conclusion"""

    def test_reads_a_last_line_that_can_be_a_conclusion(self):
        self.assertEqual(last_line_answer('x</think>\nSo the area is 12.'), '12')

    def test_reads_none_for_a_last_line_that_stops_mid_expression(self):
        self.assertIsNone(last_line_answer('x</think>\nSo $S = z + z^{-1'))

    def test_reads_none_when_the_thinking_never_closes(self):
        self.assertIsNone(last_line_answer('still thinking, never closed'))

    def test_reads_the_last_line_of_the_whole_output_when_thinking_is_off(self):
        self.assertEqual(last_line_answer('The base is 4.\nSo the area is 12.', no_think=True), '12')


class JudgeTest(unittest.TestCase):
    """judge(): (right, the answer compared, the line it came from) for a gold answer and a generation"""

    # ---- a box exists: only it is read ----
    def test_a_right_box_is_right_by_the_boxed_rule(self):
        self.assertEqual(judge('7', 'Let me work through this.</think>\n**Final Answer:**\nThe final answer is $\\boxed{7}$.'),
                         (True, '7', 'boxed'))

    def test_the_box_decides_against_a_right_marked_line(self):
        text = 'Let me work through this.</think>\nThe answer is 8.\nThe final answer is $\\boxed{7}$'
        self.assertEqual(judge('8', text), (False, '7', 'boxed'))

    def test_a_box_nested_two_brace_levels_deep_is_read(self):
        # read to its matching brace at any depth since 2026-11-26 (ISSUES #30)
        text = 'Let me work through this.</think>\n**Final Answer:**\nThe final answer is $\\boxed{\\frac{\\sqrt{3}}{3}}$.'
        self.assertEqual(judge('\\frac{\\sqrt{3}}{3}', text), (True, '\\frac{\\sqrt{3}}{3}', 'boxed'))

    # ---- no box: the last line carrying an answer marker ----
    def test_a_right_marked_line_is_right_by_the_marked_line_rule(self):
        self.assertEqual(judge('4', 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.'),
                         (True, '4', 'marked line'))

    def test_a_right_marked_line_rescues_the_answer_from_a_stray_last_line(self):
        # the marked line is kept because of this: a stray line that follows the answer is not read
        text = 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.\nIf a student is in MATH'
        self.assertEqual(judge('4', text), (True, '4', 'marked line'))

    def test_a_right_marked_line_beats_a_last_line_that_is_read_and_wrong(self):
        # the other half of the union: the last line can be a conclusion and still not be the answer
        text = 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.\nWait, let me double-check this answer.'
        self.assertEqual(last_line_answer(text), 'answer')
        self.assertEqual(judge('4', text), (True, '4', 'marked line'))

    def test_the_marked_line_decides_when_both_readings_are_right(self):
        text = 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.\nSo the number of sides is 4.'
        self.assertEqual(judge('4', text), (True, '4', 'marked line'))

    # ---- no box, and the marked line is wrong: the section's last line ----
    def test_a_right_last_line_is_right_by_the_last_line_rule_when_the_marked_line_is_wrong(self):
        # the case of ISSUES #31: the model's real conclusion stands on a later line without a marker
        text = 'Let me work through this.</think>\n**Answer:**\n$x$ is 256\nThe value of $x$ is **4**.'
        self.assertEqual(judge('4', text), (True, '4', 'last line'))

    def test_a_right_last_line_is_right_when_no_line_carries_a_marker(self):
        text = ('Let me work through this.</think>\nThere are 4 lines through these points, since two pairs of\n'
                'them are parallel.\nSo the number of lines is **4**.')
        self.assertEqual(judge('4', text), (True, '4', 'last line'))

    # ---- neither reading is right: the rule still names the line the compared answer came from ----
    def test_a_wrong_marked_line_gives_its_reading_and_names_the_marked_line(self):
        # Changed 2026-11-26 (second review): `rule` was None whenever nothing was right. It now names the
        # line the compared answer came from, right or wrong
        self.assertEqual(judge('5', 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.'),
                         (False, '4', 'marked line'))

    def test_a_wrong_last_line_without_a_marker_gives_its_reading_and_names_the_last_line(self):
        # Changed 2026-11-26 (second review): as above, the rule was None
        self.assertEqual(judge('5', 'Let me work through this.</think>\nSo the area is 12.'), (False, '12', 'last line'))

    def test_when_both_lines_are_wrong_the_compared_answer_is_the_marked_lines(self):
        text = 'Let me work through this.</think>\n**Answer:**\nThe answer is 4.\nSo the area is 12.'
        self.assertEqual(judge('5', text), (False, '4', 'marked line'))

    def test_a_generation_with_nothing_to_read_gives_none_twice(self):
        # no section, or only a line that cannot be a conclusion and no marker
        for text in ('Still thinking, never closed', 'Cut off here</think>', 'Cut off here</think>\nSo $S = z + z^{-1'):
            with self.subTest(text=text):
                self.assertEqual(judge('4', text), (False, None, None))

    # ---- reviewed 2026-11-26: a blank box is passed over ----
    def test_a_blank_box_is_passed_over_for_the_lines(self):
        # a box that quotes the instruction's empty box is not an answer: the lines are read as without a box
        for box in ('\\boxed{}', '\\boxed{ \\ }'):
            with self.subTest(box=box):
                text = 'Let me work through this.</think>\nThe answer is $' + box + '$.\nFinal answer: 4.'
                self.assertEqual(judge('4', text), (True, '4', 'marked line'))

    # ---- thinking off: the whole output is the answer section ----
    def test_with_thinking_off_a_concluding_line_needs_no_closed_think_block(self):
        self.assertEqual(judge('12', 'So the area is 12.', no_think=True), (True, '12', 'last line'))


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

    def re_score(self, texts, rows, no_think=False):
        """rescore("x", no_think) on a temporary run "x" of the rows `rows` and the generations `texts`
        ({problem id: text}); what x_rescored.json holds afterwards. The same stand-in as the test above:
        rescore() works beside itself."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "x.json").write_text(json.dumps(dict(summary=dict(name="x", mean_tokens=12), rows=rows)),
                                        encoding="utf-8")
            (tmp / "x_texts.json").write_text(json.dumps(texts), encoding="utf-8")
            with mock.patch.object(rescore, "__file__", str(tmp / "rescore.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    rescore.rescore("x", no_think)
            return json.loads((tmp / "x_rescored.json").read_text(encoding="utf-8"))

    # ---- reviewed 2026-11-26: what a re-score records besides the verdicts ----
    def test_the_summary_records_whether_thinking_was_off(self):
        # an input to every verdict, so it is on record with them
        texts = {"m0": "The answer is $7$."}       # no </think>: read only with thinking off
        for no_think in (False, True):
            with self.subTest(no_think=no_think):
                out = self.re_score(texts, [dict(id="m0", gold="7")], no_think)
                self.assertIs(out["summary"]["no_think"], no_think)
                self.assertIs(out["rows"][0]["ok_rescored"], no_think)

    def test_each_row_records_the_rule_that_decided_its_verdict(self):
        texts = {"m0": "The final answer is $\\boxed{4}$.",
                 "m1": "x</think>\n**Answer:**\nThe answer is 4.",
                 "m2": "x</think>\nThere are **4** lines.",
                 "m3": "x</think>\nSo $S = z + z^{-1"}
        out = self.re_score(texts, [dict(id=i, gold="4") for i in ("m0", "m1", "m2", "m3")])
        self.assertEqual([r["rule"] for r in out["rows"]], ["boxed", "marked line", "last line", None])
        self.assertEqual([r["rule"] for r in out["rows"]], [judge(r["gold"], texts[r["id"]])[2] for r in out["rows"]])


if __name__ == "__main__":
    unittest.main()
