"""Dependency-free MATH-500 answer scorer (replaces math-verify, broken on this
platform — see ISSUES #9). Canonical-string match OR exact numeric equivalence
via Fraction. Symmetric across arms; no subprocesses.

Usage: uv run python phase3b/scorer.py   (runs sanity tests)
"""

import re
from fractions import Fraction


def canonical(s: str) -> str:
    s = s.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "").replace("\\$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace("^{0}", "").replace("^0", "")
    s = s.replace(" ", "")
    s = s.rstrip(".")
    if s.startswith("+"):
        s = s[1:]
    return s


def to_number(s: str):
    """Exact numeric parse: ints, decimals, simple fractions a/b, \\frac{a}{b}."""
    s = canonical(s)
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
        except Exception:
            return None
        return None
    return None


def answers_match(gold: str, pred: str) -> bool:
    if pred is None:
        return False
    g, p = canonical(gold), canonical(pred)
    if g == p:
        return True
    gn, pn = to_number(g), to_number(p)
    if gn is not None and pn is not None:
        return gn == pn
    return False


if __name__ == "__main__":
    cases = [
        ("72", "72", True), ("72", "$72$", True), ("\\frac{1}{2}", "0.5", True),
        ("\\frac{1}{2}", "1/2", True), ("0.75", "\\frac{3}{4}", True),
        ("-3", "-3.0", True), ("2\\pi", "2\\pi", True), ("x=2", "2", False),
        ("(3,5)", "(3, 5)", True), ("5", "6", False), ("1,000", "1000", False),
    ]
    n_ok = 0
    for gold, pred, want in cases:
        got = answers_match(gold, pred)
        n_ok += got == want
        print(f"  {gold!r:20s} vs {pred!r:14s} -> {got} (want {want})")
    print(f"{n_ok}/{len(cases)} pass")
