"""Dependency-free MATH-500 answer scorer (replaces math-verify, broken on this
platform — see ISSUES #9). Canonical-string match OR exact numeric equivalence
via Fraction. Symmetric across arms; no subprocesses.

Third rule, 2026-11-26 (ISSUES #30). The matcher was stricter than math-verify: it rejected
answers that differed from the gold only in formatting. `answers_match` now also accepts
  - another fraction command or shorthand (`\\dfrac{33}{100}`, `\\frac43`, `\\frac 59`,
    `\\frac9{19}`) and a square root written without braces (`11\\sqrt2`);
  - a degree sign on the gold (`90^\\circ` for an answer of 90);
  - a unit written after the value (`864 \\mbox{ inches}^2`, `5.4 \\text{ cents}`);
  - thousands separators (`58,500`);
  - a variable in front of the gold (`x=5`, `y = 2x + 3`, `x \\in [-2,7]`);
  - a choice letter in parentheses (`\\text{(C)}` for C);
  - a number in another base (`52_8`), a mixed number (`137 \\frac{1}{2}`), a power (`2^{10}`);
  - tuples and fractions compared part by part (`(1.5,-13)` for `(\\frac{3}{2},-13)`);
  - a plain word that differs only in capital letters.
It only ever adds matches: the comparison of 2026-11-24 is tried first, unchanged.

Usage: uv run python phase3b/scorer.py   (runs sanity tests)
"""

import math
import re
from fractions import Fraction

# whole unit words only: a letter that could be a variable ("2m") is not a unit
_UNITS = (r"(?:degrees?|radians?|inches|feet|foot|yards?|miles?|cents?|dollars?|cm|mm|km|units?"
           r"|seconds?|minutes?|hours?|grams?|kilograms?|liters?|litres?|gallons?|pounds|ounces|grades?)")


def canonical(s: str) -> str:
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "$").replace("$", "")   # an escaped dollar first: "\$10" is 10, not "\10" (ISSUES #19)
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace("^{0}", "").replace("^0", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("+"):
        s = s[1:]
    return s


def strip_unit(s: str) -> str:
    """A unit written after a value goes: "864inches^2" is 864 (ISSUES #30). The unit must be a
    whole word: "2mmm" and "2programs" keep their letters."""
    return s[:len(s) - len(unit_of(s))] if unit_of(s) else s


def unit_of(s: str):
    """The unit a value carries at its end, as written ("inches^2"); a degree sign is the unit
    "degrees"; None when it carries none."""
    if re.search(r"(?:\^\{?\\circ\}?|°)\s*$", s):
        return "degrees"
    m = re.search(rf"(?<![A-Za-z])(?:{_UNITS})(?:\^?\{{?\d\}}?)?\s*$", s, re.IGNORECASE)
    return m.group(0) if m else None


def variable_of(s: str):
    """The variable a value is assigned to at its start ("x" in "x = 5", "x \\in [0,1]"); None."""
    m = re.match(r"^\s*([a-zA-Z])\s*(?:\\in\s*|=(?!=))", s)
    return m.group(1) if m else None


def normalise(s: str) -> str:
    """The formatting equivalents of ISSUES #30, removed on both sides alike."""
    s = s.strip()
    s = re.sub(r"\\[dtc]frac", r"\\frac", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*([0-9a-zA-Z])\s*(\{[^{}]*\})", r"\\frac{\1}\2", s)
    s = re.sub(r"\\sqrt\s*([0-9a-zA-Z])(?![a-zA-Z{])", r"\\sqrt{\1}", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace("°", "")                           # the degree character is a degree sign too
    s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    # an ordinal suffix ("12^{\mathrm{th}}", "12th") and a bracketed approximation ("(approximately 0.0287)");
    # only "th" is dropped as a plain suffix: "5st" is more likely a product than an ordinal
    s = re.sub(r"\^\{?\s*\\(?:mathrm|text|rm)\s*\{?\s*(?:th|st|nd|rd)\s*\}?\s*\}?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)\s*\^\{?\s*(?:th|st|nd|rd)\s*\}?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=\d)th\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\\quad\s*)?\(?\s*(?:\\text\s*)?\{?\s*approximately[^)]*\)?", "", s, flags=re.IGNORECASE)
    # a fraction with a square root in the denominator, rationalised ("\frac{1}{\sqrt{3}}" is \frac{\sqrt{3}}{3}")
    s = re.sub(r"\\frac\{1\}\{\\sqrt\{(\d+)\}\}", r"\\frac{\\sqrt{\1}}{\1}", s)
    # thousands separators ("58,500"); the comma of a pair in brackets ("(100,101)", "[100,101]") stays
    s = re.sub(r"(?<![\d,(\[{])(\d{1,3}(?:,\d{3})+)(?!\d)", lambda m: m.group(1).replace(",", ""), s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    # a variable in front of the value ("x=5", "y = 2x + 3", "x \in [-2,7]")
    s = re.sub(r"^[a-zA-Z]\s*(?:\\in\s*|=(?!=)\s*)(?=[\[\(\d\-\\])", "", s)
    # a choice letter in parentheses ("(C)")
    s = re.sub(r"^\(([A-Za-z])\)$", r"\1", s)
    # braces around one value ("\{5\}")
    s = re.sub(r"\\{([^{}]*)\\}", r"\1", s)
    return strip_unit(s.strip())


_POWERS = (re.compile(r"(\d+)\^(?:\{(-?\d+)\}|(-?\d+))"),          # "2^{10}", "2^10"
           re.compile(r"\((-?\d+)\)\^(?:\{(-?\d+)\}|(-?\d+))"))    # "(-2)^{2}": a negative base needs its brackets


def _power_of(s: str):
    """A power of whole numbers, or None. Above 200 bits (about 60 digits) or an exponent beyond 64
    the comparison is refused: no gold answer is that large, and a degenerate generation may try."""
    for rx in _POWERS:
        m = rx.fullmatch(s)
        if m:
            base, exp = int(m.group(1)), int(m.group(2) if m.group(2) is not None else m.group(3))
            if abs(exp) > 64 or (base and exp * math.log2(abs(base)) > 200):
                return None
            if exp >= 0:
                return Fraction(base ** exp)
            return Fraction(1, base ** (-exp)) if base else None
    return None


def to_number(s: str):
    """Exact numeric parse: ints, decimals, simple fractions a/b, \\frac{a}{b}; since 2026-11-26
    also a mixed number, a power of whole numbers and a number written in another base. (A power
    of zero is not evaluated: canonical() drops "^{0}" first, as it always did.)"""
    s = canonical(s)
    if len(s) > 50:  # guard: absurd digit strings (e.g. degenerate 0000... loops)
        return None
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return Fraction(int(s))
    m = re.fullmatch(r"-?\d*\.\d+", s)
    if m:
        return Fraction(s)
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return Fraction(a, b) if b != 0 else None
    m = re.fullmatch(r"\\frac\{(-?[^{}]+)\}\{(-?[^{}]+)\}", s)
    if m:
        try:
            a, b = to_number(m.group(1)), to_number(m.group(2))
            if a is not None and b is not None and b != 0:
                return a / b
        except (RecursionError, ZeroDivisionError, OverflowError):
            return None
        return None
    m = re.fullmatch(r"(-?\d+)\\frac\{(\d+)\}\{(\d+)\}", s)     # "137\frac{1}{2}"
    if m:
        whole, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if b:
            return Fraction(whole) + (1 if whole >= 0 else -1) * Fraction(a, b)
        return None
    if any(rx.fullmatch(s) for rx in _POWERS):                 # "2^{10}", "2^10", "(-2)^{2}"
        return _power_of(s)
    m = re.fullmatch(r"([0-9a-zA-Z]+)_\{?([0-9]+)\}?", s)        # "52_8"; "a_{16}" is a subscripted variable
    if m:
        digits, base = m.group(1), int(m.group(2))
        try:
            if not 2 <= base <= 36 or (len(digits) < 2 and not digits.isdigit()):
                return None
            return Fraction(int(digits, base))
        except ValueError:
            return None
    return None


def _brace_group(s: str, i: int):
    """The content of the brace group that starts at s[i] ('{'), and the index after it."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if not depth:
            return s[i + 1:j], j + 1
    return None, len(s)


def split_fraction(s: str):
    """(numerator, denominator) of `\\frac{a}{b}` or `a/b`, with nothing else around; else None.
    The parts must fill the whole string, so a trailing remark never goes unnoticed."""
    if s.startswith("\\frac{"):
        num, j = _brace_group(s, len("\\frac"))
        den, k = _brace_group(s, j) if num is not None and j < len(s) and s[j] == "{" else (None, j)
        return (num, den) if num and den and k == len(s) else None
    if "/" in s and "\\frac" not in s:
        parts, cur, depth = [], "", 0
        for ch in s:
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if ch == "/" and not depth:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        if len(parts) == 2 and all(parts):
            return parts
    return None


def split_tuple(s: str):
    """The parts of "(a, b, c)" split at the commas outside every bracket; None when the string is
    not one bracketed list or its brackets do not balance."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    parts, cur, depth = [], "", 0
    for ch in s[1:-1]:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        if depth < 0:
            return None
        if ch == "," and not depth:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if depth or not all(parts + [cur]):
        return None
    return [p.strip() for p in parts + [cur]]


def _match(g: str, p: str) -> bool:
    if g == p:
        return True
    gn, pn = to_number(g), to_number(p)
    if gn is not None and pn is not None:
        return gn == pn
    fg, fp = split_fraction(g), split_fraction(p)
    if fg and fp:
        return _match(fg[0], fp[0]) and _match(fg[1], fp[1])
    tg, tp = split_tuple(g), split_tuple(p)
    if tg and tp and len(tg) == len(tp):
        return all(_match(a, b) for a, b in zip(tg, tp))
    if g.isalpha() and p.isalpha():          # a word that differs only in capital letters
        return g.lower() == p.lower()
    return False


def answers_match(gold: str, pred: str) -> bool:
    """The comparison of 2026-11-24 first; then the formatting equivalents of ISSUES #30. A unit
    or an assigned variable is dropped only when the other side has none: "5 hours" is not
    "5 minutes", and "x = 5" is not "y = 5"."""
    if pred is None:
        return False
    g, p = canonical(gold), canonical(pred)
    if _match(g, p):
        return True
    gu, pu = unit_of(g), unit_of(p)
    if gu and pu and gu.lower() != pu.lower():
        return False
    gv, pv = variable_of(gold), variable_of(pred)
    if gv and pv and gv.lower() != pv.lower():
        return False
    return _match(canonical(normalise(gold)), canonical(normalise(pred)))


if __name__ == "__main__":
    cases = [
        ("72", "72", True), ("72", "$72$", True), ("\\frac{1}{2}", "0.5", True),
        ("\\frac{1}{2}", "1/2", True), ("0.75", "\\frac{3}{4}", True),
        ("-3", "-3.0", True), ("2\\pi", "2\\pi", True),
        ("(3,5)", "(3, 5)", True), ("5", "6", False),
        ("\\$10", "10", True), ("10", "$\\$10$", True), ("\\$18.90", "18.9", True),
        # third rule (ISSUES #30)
        ("x=2", "2", True), ("1,000", "1000", True),
        ("\\dfrac{33}{100}", "0.33", True), ("\\frac43", "\\frac{4}{3}", True), ("\\frac 59", "5/9", True),
        ("\\frac9{19}", "\\frac{9}{19}", True), ("11\\sqrt2", "11\\sqrt{2}", True),
        ("90^\\circ", "90", True), ("90^\\circ", "90°", True), ("864 \\mbox{ inches}^2", "864", True), ("5.4 \\text{ cents}", "5.4", True),
        ("58,500", "58500", True), ("(100,101)", "(100, 101)", True), ("(100,101)", "100101", False),
        ("y = 2x + 3", "2x+3", True), ("x \\in [-2,7]", "[-2,7]", True),
        ("\\text{(C)}", "C", True), ("(C)", "C", True), ("\\text{Navin}", "navin", True),
        ("52_8", "52_{8}", True), ("52_8", "42", True), ("137 \\frac{1}{2}", "137.5", True), ("2^{10}", "1024", True),
        ("\\left( \\frac{3}{2}, -13 \\right)", "(1.5,-13)", True), ("\\frac{\\sqrt{3}}{3}", "\\sqrt{3}/3", True),
        ("12", "12^{\\mathrm{th}}\\text{ grade}", True), ("12", "12th", True), ("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}", True),
        ("\\frac43", "\\frac{4}{3}x", False), ("\\frac{448}{15625}", "\\frac{448}{15625}\\quad(approximately0.0287)", True),
        ("2m", "2", False), ("2mmm", "2", False), ("2 programs", "2", False), ("\\frac{1}{3}", "0.33", False), ("5", "6", False), ("\\text{(C)}", "D", False),
        ("(1,2)", "(1,2,3)", False), ("2^{1000}", "1", False), ("2^{21}", "2097152", True), ("-2^{2}", "4", False),
        ("(-2)^{2}", "4", True), ("4", "5st", False), ("12", "12th grade", True),
        ("90^\\circ", "90 \\text{ radians}", False), ("90^\\circ", "90°", True), ("2^{10", "1024", False),
        ("5 \\text{ hours}", "5 \\text{ minutes}", False), ("5 \\text{ hours}", "5", True), ("x=5", "y=5", False),
        ("[100,101]", "[100101]", False), ("\\{10,100\\}", "\\{10100\\}", False), ("10", "a_{16}", False), ("10", "b_{12}", False),
        ("\\frac{448}{15625}", "\\frac{448}{15625}(\\text{approximately}0.0287)", True),
    ]
    n_ok = 0
    for gold, pred, want in cases:
        got = answers_match(gold, pred)
        n_ok += got == want
        if got != want:
            print(f"  FAIL {gold!r:36s} vs {pred!r:20s} -> {got} (want {want})")
    print(f"{n_ok}/{len(cases)} pass")
