"""Tests for phase3b/scorer.py, the answer matcher (see the scorer module docstring).
  answers_match   the gold against a predicted answer. The comparison of 2026-11-24 (canonical strings,
                  or exact numeric equivalence) is tried first, unchanged; since 2026-11-26 (the third
                  rule, ISSUES #30) the same comparison is then made on normalise()d strings, so that an
                  answer which differs from the gold only in formatting is right. It only ever adds
                  matches: every equivalence is tested next to the case that must stay wrong.
  normalise       the formatting equivalents, through answers_match();
  strip_unit      a whole unit word written after the value;
  to_number       the exact value of a mixed number, a power and a number written in another base;
  split_fraction  \\frac{a}{b}, or one top-level a/b, that fills the whole string;
  split_tuple     the parts of "(a, b, c)" split at the commas outside every bracket.
Reviewed 2026-11-26, after the first tests (the tests of it are marked "reviewed"): a unit is a whole word, and
two units (a degree sign is one) or two assigned variables are compared when both sides carry one; a power's exponent
is at most 64 in absolute value and its value at most 200 bits, a negative base needs its brackets and an exponent's
brace must close; only "th" is dropped as a plain ordinal suffix; a bracketed approximation goes without \\quad too.
The sanity list under `if __name__ == "__main__"` of scorer.py is run as it is (SanityListTest).

Standard library only. Usage (from the repository root):
  python -B -m unittest -v phase3b/test_scorer.py
"""

import contextlib
import io
import re
import runpy
import sys
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the module under test is imported by bare name, as test_rescore.py does
from scorer import answers_match, split_fraction, split_tuple, strip_unit, to_number  # noqa: E402


class AnswersMatchTest(unittest.TestCase):
    """answers_match(): (gold, predicted answer) -> right or wrong"""

    # ---- the comparison of 2026-11-24, first and unchanged ----
    def test_an_answer_the_first_comparison_accepted_is_still_accepted(self):
        # cases of the sanity list as it stood before the third rule; the first comparison still decides them
        accepted = [("72", "72"), ("72", "$72$"), ("\\frac{1}{2}", "0.5"), ("\\frac{1}{2}", "1/2"),
                    ("0.75", "\\frac{3}{4}"), ("-3", "-3.0"), ("2\\pi", "2\\pi"), ("(3,5)", "(3, 5)"),
                    ("\\$10", "10"), ("10", "$\\$10$"), ("\\$18.90", "18.9")]
        for gold, pred in accepted:
            with self.subTest(gold=gold, pred=pred):
                self.assertTrue(answers_match(gold, pred))

    def test_an_answer_of_another_value_stays_wrong(self):
        self.assertFalse(answers_match("5", "6"))

    def test_a_missing_answer_is_never_a_match(self):
        self.assertFalse(answers_match("72", None))

    def test_the_formatting_equivalents_are_removed_on_both_sides_alike(self):
        # whichever side carries the formatting, it is the same answer
        plain = [("\\dfrac{33}{100}", "0.33"), ("11\\sqrt2", "11\\sqrt{2}"), ("90^\\circ", "90"),
                 ("864 \\mbox{ inches}^2", "864"), ("12th", "12"), ("58,500", "58500"), ("x=5", "5"),
                 ("(C)", "C"), ("52_8", "42")]
        for formatted, value in plain:
            with self.subTest(formatted=formatted):
                self.assertTrue(answers_match(formatted, value))
                self.assertTrue(answers_match(value, formatted))

    # ---- another fraction command, and the shorthand without braces ----
    def test_another_fraction_command_is_the_same_fraction(self):
        for gold in ("\\dfrac{33}{100}", "\\tfrac{33}{100}", "\\cfrac{33}{100}"):
            with self.subTest(gold=gold):
                self.assertTrue(answers_match(gold, "0.33"))

    def test_fraction_shorthand_without_braces_is_the_same_fraction(self):
        for gold, pred in [("\\frac43", "\\frac{4}{3}"), ("\\frac 59", "5/9"), ("\\frac9{19}", "\\frac{9}{19}")]:
            with self.subTest(gold=gold, pred=pred):
                self.assertTrue(answers_match(gold, pred))

    def test_a_decimal_that_only_approximates_a_fraction_stays_wrong(self):
        self.assertFalse(answers_match("\\frac{1}{3}", "0.33"))

    def test_a_fraction_with_something_after_it_is_not_that_fraction(self):
        self.assertFalse(answers_match("\\frac{4}{3}x", "\\frac43"))
        self.assertFalse(answers_match("\\frac43", "\\frac{4}{3}x"))

    # ---- a square root written without braces ----
    def test_a_square_root_without_braces_is_the_same_root(self):
        self.assertTrue(answers_match("11\\sqrt2", "11\\sqrt{2}"))

    def test_another_root_stays_wrong(self):
        self.assertFalse(answers_match("11\\sqrt2", "11\\sqrt3"))

    # ---- a degree sign ----
    def test_a_degree_sign_on_the_gold_is_not_part_of_the_answer(self):
        self.assertTrue(answers_match("90^\\circ", "90"))

    def test_the_degree_character_is_a_degree_sign_too(self):
        self.assertTrue(answers_match("90^\\circ", "90\u00b0"))

    def test_another_angle_stays_wrong(self):
        self.assertFalse(answers_match("90^\\circ", "45"))

    # ---- a unit written after the value ----
    def test_a_unit_wrapped_in_a_text_command_is_dropped(self):
        self.assertTrue(answers_match("864 \\mbox{ inches}^2", "864"))
        self.assertTrue(answers_match("5.4 \\text{ cents}", "5.4"))

    def test_a_letter_after_the_value_is_not_a_unit(self):
        # "m" could be a variable; only whole unit words go
        self.assertFalse(answers_match("2m", "2"))

    def test_an_abbreviated_unit_is_not_a_whole_unit_word(self):
        self.assertFalse(answers_match("864 \\text{ in}^2", "864"))
        self.assertFalse(answers_match("30 \\text{ sec}", "30"))

    # reviewed 2026-11-26: a unit is a whole word, and two units are compared when both sides carry one
    def test_a_word_that_ends_in_a_unit_word_is_not_a_unit(self):
        for gold in ("2mmm", "2 programs"):
            with self.subTest(gold=gold):
                self.assertFalse(answers_match(gold, "2"))

    def test_two_different_units_do_not_compare(self):
        self.assertFalse(answers_match("5 \\text{ hours}", "5 \\text{ minutes}"))
        self.assertFalse(answers_match("864 \\mbox{ inches}^2", "864 \\text{ cm}^2"))

    def test_a_degree_sign_is_compared_with_a_named_unit(self):
        # 90 degrees is not 90 radians: the degree sign is the unit of the gold
        self.assertFalse(answers_match("90^\\circ", "90 \\text{ radians}"))

    def test_a_degree_sign_compares_with_the_unit_spelled_out(self):
        # the same unit on both sides: the values alone are compared
        self.assertTrue(answers_match("90^\\circ", "90 \\text{ degrees}"))
        self.assertFalse(answers_match("45^\\circ", "90 \\text{ degrees}"))

    def test_a_unit_next_to_a_bare_value_is_still_dropped(self):
        self.assertTrue(answers_match("5 \\text{ hours}", "5"))

    # ---- an ordinal suffix ----
    def test_an_ordinal_suffix_is_not_part_of_the_answer(self):
        self.assertTrue(answers_match("12", "12^{\\mathrm{th}}\\text{ grade}"))
        self.assertTrue(answers_match("12", "12th"))

    def test_another_ordinal_stays_wrong(self):
        self.assertFalse(answers_match("12", "13th"))

    def test_a_plain_th_suffix_is_dropped_with_what_follows_it(self):
        self.assertTrue(answers_match("12", "12th grade"))

    def test_another_plain_ordinal_suffix_is_not_dropped(self):
        # reviewed 2026-11-26: "5st" is more likely a product than an ordinal, so only "th" goes as a plain suffix
        for gold, pred in (("4", "5st"), ("2", "2nd"), ("3", "3rd"), ("1", "1st")):
            with self.subTest(pred=pred):
                self.assertFalse(answers_match(gold, pred))

    # ---- a bracketed approximation after the answer ----
    def test_a_bracketed_approximation_after_the_answer_is_ignored(self):
        self.assertTrue(answers_match("\\frac{448}{15625}", "\\frac{448}{15625}\\quad(approximately0.0287)"))
        self.assertTrue(answers_match("\\frac{448}{15625}", "\\frac{448}{15625} \\quad (\\text{approximately } 0.0287)"))

    def test_the_approximation_alone_is_not_the_exact_answer(self):
        self.assertFalse(answers_match("\\frac{448}{15625}", "0.0287"))

    def test_a_bracketed_approximation_is_ignored_without_a_quad_too(self):
        # reviewed 2026-11-26: it used to go only after \quad
        self.assertTrue(answers_match("\\frac{448}{15625}", "\\frac{448}{15625}(\\text{approximately}0.0287)"))

    # ---- a fraction by a square root, rationalised ----
    def test_a_fraction_by_a_square_root_is_rationalised(self):
        self.assertTrue(answers_match("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}"))

    def test_a_fraction_by_another_square_root_stays_wrong(self):
        self.assertFalse(answers_match("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{2}}"))

    # ---- thousands separators ----
    def test_thousands_separators_are_ignored(self):
        self.assertTrue(answers_match("58,500", "58500"))
        self.assertTrue(answers_match("1,000", "1000"))

    def test_another_number_stays_wrong(self):
        self.assertFalse(answers_match("58,500", "5850"))

    def test_the_comma_of_a_pair_in_brackets_is_not_a_thousands_separator(self):
        self.assertFalse(answers_match("(100,101)", "100101"))
        self.assertFalse(answers_match("(100,101)", "(100101)"))     # the pair is not one number of six digits
        self.assertTrue(answers_match("(100,101)", "(100, 101)"))    # the same pair, spaced

    def test_a_comma_after_an_opening_bracket_is_not_a_thousands_separator(self):
        # reviewed 2026-11-26: "[100,101]" used to read as one number
        self.assertFalse(answers_match("[100,101]", "[100101]"))
        self.assertFalse(answers_match("\\{10,100\\}", "\\{10100\\}"))

    # ---- a variable in front of the value ----
    def test_a_variable_in_front_of_the_value_is_dropped(self):
        for gold, pred in [("x=5", "5"), ("y = 2x + 3", "2x+3"), ("x \\in [-2,7]", "[-2,7]")]:
            with self.subTest(gold=gold, pred=pred):
                self.assertTrue(answers_match(gold, pred))

    def test_the_expression_behind_the_variable_stays_the_answer(self):
        self.assertFalse(answers_match("y = 2x + 3", "3"))

    # reviewed 2026-11-26: two variables are compared when both sides carry one
    def test_two_different_variables_do_not_compare(self):
        self.assertFalse(answers_match("x=5", "y=5"))

    def test_the_same_variable_in_another_case_compares(self):
        self.assertTrue(answers_match("X=5", "x=5"))

    # ---- a choice letter in parentheses ----
    def test_a_choice_letter_in_parentheses_is_the_letter(self):
        self.assertTrue(answers_match("\\text{(C)}", "C"))
        self.assertTrue(answers_match("(C)", "C"))

    def test_another_choice_letter_stays_wrong(self):
        self.assertFalse(answers_match("\\text{(C)}", "D"))

    # ---- a plain word that differs only in capital letters ----
    def test_a_word_that_differs_only_in_capital_letters_is_the_same_word(self):
        self.assertTrue(answers_match("\\text{Even}", "even"))

    def test_another_word_stays_wrong(self):
        self.assertFalse(answers_match("\\text{Even}", "odd"))

    # ---- braces round one value ----
    def test_braces_round_one_value_are_dropped(self):
        self.assertTrue(answers_match("\\{5\\}", "5"))

    def test_another_value_in_braces_stays_wrong(self):
        self.assertFalse(answers_match("\\{5\\}", "\\{6\\}"))

    # ---- a mixed number ----
    def test_a_mixed_number_is_its_value(self):
        self.assertTrue(answers_match("137 \\frac{1}{2}", "137.5"))

    def test_the_whole_part_of_a_mixed_number_is_not_its_value(self):
        self.assertFalse(answers_match("137 \\frac{1}{2}", "137"))

    # ---- a power: the exponent at most 64 in absolute value, and at most 200 bits (reviewed 2026-11-26) ----
    def test_a_power_is_its_value(self):
        self.assertTrue(answers_match("2^{10}", "1024"))

    def test_a_power_with_an_exponent_of_sixty_four_is_still_evaluated(self):
        self.assertTrue(answers_match("2^{64}", "18446744073709551616"))

    def test_a_negative_exponent_is_evaluated_too(self):
        self.assertTrue(answers_match("2^{-1}", "0.5"))

    def test_a_power_with_an_exponent_above_sixty_four_is_not_evaluated(self):
        # Changed 2026-11-26 (second review): the cap was 20 in absolute value, and this test refused 2^{21}.
        # The cap is 64 now, so 2^{21} is its value; beyond the cap the answer is not a number to compare
        self.assertTrue(answers_match("2^{21}", "2097152"))
        for gold, pred in (("2^{65}", "36893488147419103232"), ("2^{65}", "1"), ("2^{-65}", "0")):
            with self.subTest(gold=gold, pred=pred):
                self.assertFalse(answers_match(gold, pred))

    def test_a_power_with_an_exponent_of_a_thousand_matches_no_other_value(self):
        for pred in ("1", "1024", "2^{10}"):
            with self.subTest(pred=pred):
                self.assertFalse(answers_match("2^{1000}", pred))

    def test_a_negative_base_needs_its_brackets(self):
        # -2^{2} is -(2^{2}): the minus is left outside, and the rest is not a number to compare
        self.assertFalse(answers_match("4", "-2^{2}"))

    def test_a_negative_base_in_brackets_is_its_value(self):
        self.assertTrue(answers_match("4", "(-2)^{2}"))
        self.assertTrue(answers_match("-8", "(-2)^{3}"))

    # ---- a number written in another base ----
    def test_a_number_in_another_base_is_its_value(self):
        self.assertTrue(answers_match("52_8", "42"))
        self.assertTrue(answers_match("4210_{5}", "555"))

    def test_the_digits_of_a_number_in_another_base_are_not_a_decimal_number(self):
        self.assertFalse(answers_match("52_8", "52"))

    def test_a_digit_the_base_cannot_hold_is_not_a_number(self):
        self.assertFalse(answers_match("52_4", "6"))

    def test_a_subscripted_single_letter_is_not_a_number_in_another_base(self):
        # reviewed 2026-11-26: "a_{16}" is a variable with a subscript, not 10
        self.assertFalse(answers_match("10", "a_{16}"))
        self.assertFalse(answers_match("11", "b_{12}"))

    # ---- the parts of a fraction or a tuple, compared part by part ----
    def test_a_slash_fraction_is_compared_part_by_part(self):
        self.assertTrue(answers_match("\\frac{\\sqrt{3}}{3}", "\\sqrt{3}/3"))

    def test_a_slash_fraction_that_differs_in_one_part_stays_wrong(self):
        self.assertFalse(answers_match("\\frac{\\sqrt{3}}{3}", "\\sqrt{3}/2"))

    def test_a_fraction_inside_a_tuple_is_compared_part_by_part(self):
        self.assertTrue(answers_match("(\\frac{3}{2},-13)", "(1.5,-13)"))

    def test_a_tuple_inside_a_tuple_is_compared_part_by_part(self):
        self.assertTrue(answers_match("((1,2),(3,\\frac{1}{2}))", "((1,2),(3,0.5))"))

    def test_a_tuple_that_differs_in_one_part_stays_wrong(self):
        self.assertFalse(answers_match("(\\frac{3}{2},-13)", "(1.5,-12)"))

    def test_a_tuple_with_another_number_of_parts_stays_wrong(self):
        self.assertFalse(answers_match("(1,2)", "(1,2,3)"))

    def test_a_tuple_in_another_order_stays_wrong(self):
        self.assertFalse(answers_match("(1,2)", "(2,1)"))

    def test_an_approximation_inside_a_tuple_stays_wrong(self):
        self.assertFalse(answers_match("(\\frac{1}{2},\\frac{1}{3})", "(0.5,0.33)"))


class StripUnitTest(unittest.TestCase):
    """strip_unit(): a whole unit word written after a value"""

    def test_a_whole_unit_word_after_the_value_is_dropped(self):
        self.assertEqual(strip_unit("864inches^2"), "864")
        self.assertEqual(strip_unit("12grade"), "12")

    def test_an_exponent_on_the_unit_goes_with_it(self):
        self.assertEqual(strip_unit("864inches^{2}"), "864")

    def test_a_letter_after_the_value_stays(self):
        self.assertEqual(strip_unit("2m"), "2m")

    def test_an_abbreviated_unit_stays(self):
        self.assertEqual(strip_unit("864in^2"), "864in^2")
        self.assertEqual(strip_unit("30sec"), "30sec")

    def test_a_word_that_ends_in_a_unit_word_stays(self):
        # reviewed 2026-11-26: the unit must start a word ("2mmm" used to lose its last two letters)
        self.assertEqual(strip_unit("2mmm"), "2mmm")
        self.assertEqual(strip_unit("2programs"), "2programs")


class ToNumberTest(unittest.TestCase):
    """to_number(): the exact value of an answer written as a number"""

    # ---- a mixed number ----
    def test_reads_a_mixed_number_as_its_value(self):
        self.assertEqual(to_number("137 \\frac{1}{2}"), Fraction(275, 2))

    def test_a_negative_mixed_number_is_negative(self):
        self.assertEqual(to_number("-137\\frac{1}{2}"), Fraction(-275, 2))

    # ---- a power: the exponent at most 64 in absolute value, and at most 200 bits (reviewed 2026-11-26) ----
    def test_reads_a_power_as_its_value(self):
        self.assertEqual(to_number("2^{10}"), 1024)
        self.assertEqual(to_number("2^10"), 1024)

    def test_a_negative_exponent_gives_a_fraction(self):
        self.assertEqual(to_number("2^{-1}"), Fraction(1, 2))

    def test_an_exponent_of_sixty_four_in_either_sign_is_still_evaluated(self):
        self.assertEqual(to_number("2^{64}"), 18446744073709551616)
        self.assertEqual(to_number("2^{-64}"), Fraction(1, 18446744073709551616))

    def test_an_exponent_above_sixty_four_in_absolute_value_is_refused(self):
        # Changed 2026-11-26 (second review): the cap was 20 in absolute value, and 2^{21} and 2^{-21} were
        # refused here. The cap is 64 now: both are evaluated, and a thousand is beyond it
        self.assertEqual(to_number("2^{21}"), 2097152)
        self.assertEqual(to_number("2^{-21}"), Fraction(1, 2097152))
        for s in ("2^{65}", "2^{-65}", "2^{1000}", "2^{-1000}"):
            with self.subTest(s=s):
                self.assertIsNone(to_number(s))

    def test_a_power_of_exactly_two_hundred_bits_is_still_evaluated(self):
        self.assertEqual(to_number("16^{50}"), 16 ** 50)       # 16^{50} is 2^{200}

    def test_a_power_of_more_than_two_hundred_bits_is_refused(self):
        # an exponent within the cap: the size of the value refuses it
        self.assertIsNone(to_number("16^{51}"))                # 2^{204}
        self.assertIsNone(to_number("999999999^{64}"))

    def test_a_negative_exponent_of_zero_is_refused(self):
        self.assertIsNone(to_number("0^{-1}"))

    def test_a_negative_base_without_its_brackets_is_not_a_number(self):
        # -2^{2} is -(2^{2}): the minus is left outside
        self.assertIsNone(to_number("-2^{2}"))

    def test_a_negative_base_in_brackets_is_its_value(self):
        self.assertEqual(to_number("(-2)^{2}"), 4)
        self.assertEqual(to_number("(-2)^{3}"), -8)

    def test_an_exponent_whose_brace_never_closes_is_not_a_number(self):
        # "2^{10" is a generation cut off inside the exponent: a brace that opens must close
        self.assertIsNone(to_number("2^{10"))
        self.assertEqual(to_number("2^{10}"), 1024)
        self.assertEqual(to_number("2^10"), 1024)

    def test_a_bracketed_base_whose_exponent_brace_never_closes_is_not_a_number(self):
        self.assertIsNone(to_number("(-2)^{2"))
        self.assertEqual(to_number("(-2)^{2}"), 4)
        self.assertEqual(to_number("(-2)^2"), 4)

    # ---- a number written in another base ----
    def test_reads_a_number_in_another_base_as_its_value(self):
        self.assertEqual(to_number("52_8"), 42)
        self.assertEqual(to_number("52_{8}"), 42)
        self.assertEqual(to_number("4210_{5}"), 555)

    def test_a_digit_the_base_cannot_hold_is_refused(self):
        self.assertIsNone(to_number("52_4"))

    def test_a_base_outside_2_to_36_is_refused(self):
        for s in ("11_1", "10_37"):
            with self.subTest(s=s):
                self.assertIsNone(to_number(s))

    def test_a_subscripted_single_letter_is_not_a_number_in_a_base(self):
        self.assertIsNone(to_number("a_{16}"))
        self.assertIsNone(to_number("b_{12}"))


class SplitFractionTest(unittest.TestCase):
    """split_fraction(): numerator and denominator of a fraction that fills the whole string, else None"""

    def test_a_fraction_command_is_split_into_its_two_parts(self):
        self.assertEqual(list(split_fraction("\\frac{4}{3}")), ["4", "3"])

    def test_the_parts_of_a_fraction_command_may_hold_braces(self):
        self.assertEqual(list(split_fraction("\\frac{\\sqrt{3}}{3}")), ["\\sqrt{3}", "3"])

    def test_one_top_level_slash_is_split_into_its_two_parts(self):
        self.assertEqual(list(split_fraction("4/3")), ["4", "3"])
        self.assertEqual(list(split_fraction("\\sqrt{3}/3")), ["\\sqrt{3}", "3"])

    def test_a_slash_inside_braces_is_not_top_level(self):
        self.assertIsNone(split_fraction("{a/b}"))

    def test_a_second_top_level_slash_leaves_the_string_unsplit(self):
        self.assertIsNone(split_fraction("1/2/3"))

    def test_text_after_a_fraction_command_leaves_it_unsplit(self):
        self.assertIsNone(split_fraction("\\frac{4}{3}x"))
        self.assertIsNone(split_fraction("\\frac{1}{2}/3"))

    def test_text_before_a_fraction_command_leaves_it_unsplit(self):
        self.assertIsNone(split_fraction("x\\frac{4}{3}"))

    def test_an_unclosed_fraction_command_is_not_split(self):
        self.assertIsNone(split_fraction("\\frac{4}{"))
        self.assertIsNone(split_fraction("\\frac{4}"))

    def test_an_empty_part_is_not_a_fraction(self):
        self.assertIsNone(split_fraction("a/"))
        self.assertIsNone(split_fraction("/2"))
        self.assertIsNone(split_fraction("\\frac{}{2}"))     # reviewed 2026-11-26: this used to give ("", "2")
        self.assertIsNone(split_fraction("\\frac{2}{}"))

    def test_a_string_without_a_fraction_is_not_split(self):
        self.assertIsNone(split_fraction("3"))
        self.assertIsNone(split_fraction("x + 1"))


class SplitTupleTest(unittest.TestCase):
    """split_tuple(): the parts of one bracketed list, split outside every bracket, else None"""

    def test_a_tuple_is_split_at_its_commas(self):
        self.assertEqual(split_tuple("(3, 5)"), ["3", "5"])
        self.assertEqual(split_tuple("(a, b, c)"), ["a", "b", "c"])

    def test_a_comma_inside_a_bracket_is_not_a_separator(self):
        self.assertEqual(split_tuple("((1,2),(3,4))"), ["(1,2)", "(3,4)"])
        self.assertEqual(split_tuple("(\\frac{1}{2}, 3)"), ["\\frac{1}{2}", "3"])

    def test_a_tuple_that_does_not_close_is_refused(self):
        self.assertIsNone(split_tuple("(1,2"))
        self.assertIsNone(split_tuple("(1,(2)"))

    def test_a_bracket_closed_once_too_often_is_refused(self):
        self.assertIsNone(split_tuple("(1,2))"))

    def test_an_empty_part_is_refused(self):
        for s in ("(1,,2)", "(1,2,)", "()"):
            with self.subTest(s=s):
                self.assertIsNone(split_tuple(s))

    def test_a_string_that_is_not_one_bracketed_list_is_refused(self):
        for s in ("3, 5", "[1,2]", "(1,2) (3,4)"):
            with self.subTest(s=s):
                self.assertIsNone(split_tuple(s))


class SanityListTest(unittest.TestCase):
    """The sanity list under `if __name__ == "__main__"` of scorer.py, run by the module's own loop"""

    def test_every_case_of_the_sanity_list_scores_as_intended(self):
        # the list is the specification of the third rule; it is left in scorer.py and run from there
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            runpy.run_path(str(HERE / "scorer.py"), run_name="__main__")
        self.assertEqual([line for line in printed.getvalue().splitlines() if line.lstrip().startswith("FAIL")], [],
                         "a case of the sanity list in scorer.py scores otherwise than it intends")
        counted = re.search(r"^(\d+)/(\d+) pass$", printed.getvalue(), re.MULTILINE)
        self.assertIsNotNone(counted, "the sanity run of scorer.py printed no 'N/N pass' line")
        self.assertGreaterEqual(int(counted.group(2)), 48)     # the list as it stands; it must not have gone short


if __name__ == "__main__":
    unittest.main()
