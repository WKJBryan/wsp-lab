r"""Phase-3c rescoring v2: format-aware extraction (ISSUES #10), fixed after audit.

Extraction order (identical for both arms, from saved generations):
  1. the last complete \\boxed{...} anywhere;
  2. else, in the post-</think> section: the LAST line carrying an answer marker
     ("answer:", "final answer", "answer is", "value is", "coordinates are"), and the section's
     last line when it can be a conclusion — the answer is right if either is right (third rule);
     sliced at the marker (length-preserving view so indices are valid);
     empty payload -> next non-empty line;
  3. else the section's last line, when it can be a conclusion.
Payload cleanup: \\text{X}->X, strip $..$, \\[ \\], bold, trailing period,
take the part after the last '='; then prefer the last $math$ span, else the
last number-ish token. Thinking that never closes leaves no answer section and
so no answer — unless a \\boxed{} stands inside it (step 1 reads the whole text).

First correction, 2026-11-24 (ISSUES #19). Two defects cost 3-5 points on every arm
that finishes, and were fixed identically for all arms:
  a. A $math$ span that ENDS the payload is read before any '$' is stripped. The
     trailing period is removed first, which makes the span's closing '$' the last
     character; the old code then stripped it and the span was lost ("... is
     $-2 + 7i$." was scored as "7"; "$$5r^5$$" as "5"). An escaped \$ inside the
     span is content, so "$\$18.90$" reads as \$18.90. A span that does not end
     the payload is read by the first checker's rule (a2): the span itself,
     unless the sentence states its value.
  c. A stray </think> token inside the answer section is removed first. In
     stop-token-boosted arms the model emits it again on the answer line.
Not changed: the split at '=' (a gold such as "x=5" is still compared with "5").

Second correction, 2026-11-24 (ISSUES #19), from real saved answers.
  a2. A math span followed by nothing but a linking word and a plain value states that value:
      "The value of $x$ is **83**" is 83, not x. The first correction still read the span.
      A span followed by anything else stays the answer ("$\frac{13}{15}$ of the pie";
      "is $\sqrt{5}$ meters").
  b2. Spans are searched before any '$' is stripped, so a span that opens the answer survives
      a unit after it ("$\frac{16}{27}$ gallons").
  e. An ordinal span is its number ("the $12^{th}$ grade").
  d. An escaped dollar is a dollar sign: "$\$10$" is 10 and also matches a gold "\$10"
      (phase3b/scorer.py, canonical).

Measured answer by answer on the 20,218 saved generations of phases 3c (the fifteen valid
500-problem runs), 8, 11 and 5, under four rules:
                                   right answers
  the rule as first committed         4,681   (the first-reported scores)
  the first re-score of 2026-11-23/24 5,226
  the first correction as last refined 5,275
  this checker                        5,795
This checker against the committed rule: 1,125 answers turn right and 11 turn wrong. Four of
the 11 were accidental hits of the old rule (it took a digit out of text that was not an
answer); seven are misreads that remain, listed in ISSUES #19. Against the first re-score:
570 right, 1 wrong (a false positive of that rule); math-verify, given only the last line of
each, agrees with 566 of the 570. Against the first correction as last refined: 520 right,
none wrong.

Third rule, 2026-11-26 (ISSUES #31 and #30), from the measurements recorded there and
reviewed twice before use.
  a3. An answer without \boxed{} is right when either reading is right: the last line carrying
      an answer marker (step 2, the previous rule), or the section's last line (step 3) when that
      line can be a conclusion (`concludes`). The previous rule read only the marked line; when
      the model's real conclusion stood on a later line without a marker, a correct answer was
      scored wrong. The marked line is kept because it rescues answers from stray trailing lines.
      A line that stops mid-expression, or is a heading or a list number, is not read: the
      previous rule took its last digit for an answer, and 20 times that digit equalled the gold
      by accident. `judge` returns (right, answer, rule); `rule` names the line the compared
      answer came from ("boxed", "marked line", "last line") or None when nothing was read.
      `extract_answer` still returns that one string for callers that want it without the verdict.
  b3. A box is read to its matching brace at any depth, in one pass. A box nested two brace
      levels deep (\boxed{\frac{\sqrt{3}}{3}}) used to be missed; a box that never closes and a
      blank box ("\boxed{}", quoted from the instruction) are passed over.
  d3. The cleanup of a payload: a bold value keeps its closing stars with or without a period
      ("is **4**" used to read "x" when no period followed); opening stars always go, and a bold
      label in front of a value goes with it ("**Answer:** **(E)**" is (E)); a period, stars that
      close nothing and a bracket that nothing opened go in any order ("\frac{3840}{289}.)"), but
      LaTeX's "\)" stays and a list keeps its brackets ("(3,5)", "[2, 5)").
  c3. The matcher accepts the formatting equivalents math-verify accepts (phase3b/scorer.py).

Measured answer by answer on the 23,718 saved answers (the 22,718 above and the two phase-14
finals) before anything was re-scored, and again by a second script: 943 turn right, 21 turn
wrong. Of the 21, 20 were accidental hits of the previous rule on text cut off at the cap or
on a heading; one is a model that boxed one value inside its thinking and concluded with
another (phase8/b1 m283: a box is read first, and stays the answer). math-verify accepts 933
of the 943 on the very string the checker accepted; the other ten are right by hand (a number
given in another base or as a decimal where the gold is in another base, markdown residue,
a matrix math-verify cannot read). By rule: matcher 585, last line 261, last line and matcher
85, box read at any depth 12.

Usage: uv run python phase3c/rescore.py <run_name> [<run_name> ...] [--no-think]
Reads <run_name>_texts.json, or <run_name>_texts.json.gz when that is absent.
"""

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3b"))
from runcheck import check_run  # noqa: E402
from scorer import answers_match, canonical  # noqa: E402

# identifies the checker that produced a score: every re-scored file records it. Covers the
# extractor and the matcher it uses; hashes the text with newlines normalised, so a Windows and a
# Linux checkout stamp alike. (In the phase-14 sandbox scorer.py sits next to this file.)
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


_HERE = Path(__file__).resolve().parent
_SCORER = next(p for p in (_HERE.parent / "phase3b" / "scorer.py", _HERE / "scorer.py") if p.exists())
CHECKSUM = hashlib.sha256((_digest(Path(__file__)) + "|" + _digest(_SCORER)).encode("utf-8")).hexdigest()[:12]
RESCORED_WITH = f"phase3c/rescore.py+phase3b/scorer.py#{CHECKSUM} (ISSUES #19, #30, #31)"

MARKER_RE = re.compile(
    r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should",
    re.IGNORECASE,
)
STAR_VIEW = str.maketrans({"*": " "})
SPAN = re.compile(r"\$((?:\\\$|[^$])+)\$")             # a $math$ span; an escaped \$ is content
END_SPAN = re.compile(r"\$((?:\\\$|[^$])+)\$\$*\s*$")   # ... and it ends the payload ($$...$$ too)
# what follows the last span when the sentence states its value: at most four words, a linking
# word ("is", "is equal to", "equals"; "is exactly"), then a number, a fraction or \frac{a}{b},
# and nothing else ("is **83**", "expressed as a decimal is 4.5", "equals $\frac{1}{2}$")
STATED_VALUE = re.compile(
    r"^(?:[a-z]+(?:\s+[a-z]+){0,3}\s+)?(?:is\s+equal\s+to|is(?:\s+exactly)?|equals|equal\s+to)\s*:?\s*"
    r"\$?\s*(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?|\\d?frac\{[^{}]*\}\{[^{}]*\})\s*\$?$",
    re.IGNORECASE)
# an ordinal in math mode is its number ("the $12^{th}$ grade" answers 12)
ORDINAL = re.compile(r"^(-?\d+)\^\{(?:\\text\{)?(st|nd|rd|th)\}?\}$", re.IGNORECASE)


def view(s: str) -> str:
    """Length-preserving lowercase, stars blanked — indices valid on the original."""
    return s.lower().translate(STAR_VIEW)


def clean(seg: str) -> str | None:
    seg = seg.strip()
    seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
    seg = seg.replace("\\[", "").replace("\\]", "")
    if "=" in seg:
        seg = seg.split("=")[-1]
    # a bold label in front of the value reads with the value ("**Answer:** **(E)**", "**Result:**
    # 326.5"); trailing punctuation then goes, in any order: a period; stars that close no bold
    # value in the payload ("**$y = \frac{4}{3}$**." once split at "="); a closing bracket that
    # nothing opened ("\frac{3840}{289}.)"), but not LaTeX's "\)"; a list keeps its brackets
    # ("(3,5)", "[2, 5)"). A bold value keeps its closing stars ("is **4**" with or without a
    # period: the first two rules stripped them and read "x").
    seg = re.sub(r"^\*\*[^*\n]{1,80}:\*\*\s*", "", seg.strip())
    seg = seg.lstrip("*").rstrip()
    opens, closes = seg.count("(") + seg.count("["), seg.count(")") + seg.count("]")
    end = len(seg)
    while end:
        ch = seg[end - 1]
        if ch in " .":
            end -= 1
        elif ch in ")]":
            if ch == ")" and end >= 2 and seg[end - 2] == "\\":
                break                                    # LaTeX's closing bracket, "\)"
            if closes > opens:
                end -= 1
                closes -= 1                              # a bracket that nothing opened
            else:
                break
        elif ch == "*":
            start = end
            while start and seg[start - 1] == "*":
                start -= 1
            if seg[:start].count("**") % 2:
                break                                  # these stars close a bold value: keep them
            end = start                              # stars that close nothing
        else:
            break
    seg = seg[:end].rstrip()
    # 1. bolded emphasis near the end ("the value is **83**")
    bolds = re.findall(r"\*\*([^*]+)\*\*", seg)
    if bolds:
        return clean_inner(bolds[-1])
    return clean_inner(seg)


def clean_inner(seg: str) -> str | None:
    seg = seg.strip()
    final = END_SPAN.search(seg)          # a math span that ends the payload, before any '$' is stripped
    if final:
        return canonical(final.group(1))
    if not seg:
        return None
    # spans are found before any '$' is stripped, so a span that opens the payload survives
    # ("$\frac{16}{27}$ gallons" was read as "\frac{16}{27}gallons")
    spans = list(SPAN.finditer(seg))
    if spans:
        after = seg[spans[-1].end():].strip().strip("*").strip().rstrip(".").strip()
        stated = STATED_VALUE.match(after)
        if stated:                      # the sentence states the value of the thing in the span
            return canonical(stated.group(1).replace(",", ""))
        last = spans[-1].group(1)
        ordinal = ORDINAL.match(last)
        return canonical(ordinal.group(1) if ordinal else last)   # the first checker's rule: any span, the last one
    seg = seg.strip("$").strip()        # a payload wrapped in one pair of dollars, or their leftovers
    if not seg:
        return None
    if "\\" in seg:  # LaTeX payload — trust it whole
        return canonical(seg)
    if re.search("[a-zA-Z]", seg):  # prose remainder — last number, else last word
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
        if nums:
            return nums[-1]
        words = re.findall(r"[A-Za-z]+", seg)
        return canonical(words[-1]) if words else None
    return canonical(seg)


_BOX_OR_BRACE = re.compile(r"\\boxed\{|[{}]")
_BLANK = re.compile(r"\\[ ,;:!]|\\q?uad|~|\s")


def _blank(content: str) -> bool:
    """A box that holds nothing but spacing ("", "\\ ", "~") quotes the instruction's empty box."""
    return not _BLANK.sub("", content)


def last_boxed(text: str):
    """The content of the last complete \\boxed{...} of a generation, read to its matching brace at
    any depth (ISSUES #30); None when there is none. A box that never closes (text cut off inside
    it) and a box that is blank ("\\boxed{}", quoted from the instruction) are not answers and are
    passed over. Of two boxes one inside the other the inner one is the answer: it is the last to
    open. One pass over the text, so a generation that repeats "\\boxed{" a thousand times costs
    no more than any other."""
    last, last_start, stack, depth = None, -1, [], 0
    for m in _BOX_OR_BRACE.finditer(text):
        token = m.group(0)
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if stack and stack[-1][1] == depth + 1:        # this brace closes the innermost open box
                start, _ = stack.pop()
                content = text[start:m.start()]
                if not _blank(content) and start > last_start:
                    last, last_start = content, start
        else:
            depth += 1
            stack.append((m.end(), depth))
    return last


def answer_lines(text: str, no_think: bool = False):
    """The non-empty lines of the answer section: after </think>, or the whole output when
    thinking is off (phase 8, phase 11). A generation that never closes its thinking has none.
    A stray </think> token inside the answer section is not answer text (defect c)."""
    if no_think:
        answer = text
    elif "</think>" in text:
        answer = text.split("</think>", 1)[1]
    else:
        return []
    lines = [l.strip().replace("</think>", " ").strip() for l in answer.splitlines()]
    return [l for l in lines if l]


def marked_line(lines):
    """The last line carrying an answer marker, sliced at the marker (length-preserving view so
    indices are valid); empty payload -> next non-empty line. None when no line carries one."""
    best = None
    for i, l in enumerate(lines):
        lv = view(l)
        m = MARKER_RE.search(lv)
        if m:
            payload = lv[m.end():].lstrip(":").strip()
            if not payload and i + 1 < len(lines):
                payload = view(lines[i + 1]).lstrip(":").strip()
            c = clean(payload)
            if c:
                best = c
    return best


def concludes(line: str) -> bool:
    """Whether a line can be a model's conclusion. It is one bare value ("90", "(C)"), or it ends
    a sentence (".", "!", "?"), closes a bracket, a math span or a bold value, or carries a label
    and a value ("**Total ways:** 3", "Result: 326.5"). A line that stops mid-expression
    ("f(1,", "= 1.5 \\cdot", "**Step 2. Simplify the equation to") is a fragment cut off at the
    cap or a stray remark: its numbers are not an answer. A heading or list number is not one
    either ("**Step 1.", "Step 2.", "3)")."""
    s = line.rstrip()
    if re.search(r"[,(\[{;=+\-/^\\]$", s) or re.match(r"\W*(?:step|case|method|part)\s*\d", s, re.IGNORECASE) \
            or re.match(r"\**\s*\d+\)$", s):
        return False
    if not re.search(r"\s", s) or re.search(r"[.!?)\]}$*]\s*$", s):
        return True
    return bool(re.search(r"(?:\*\*[^*]+\*\*|[^:*]{1,40}):\s*[*:\s]*[^\s:*]*[\w)\]}$]\s*$", s))


def last_line_answer(text: str, no_think: bool = False):
    """The section's last line read as an answer, when it can be a conclusion; else None."""
    lines = answer_lines(text, no_think)
    return clean(lines[-1]) if lines and concludes(lines[-1]) else None


def extract_answer(text: str, no_think: bool = False):
    """One answer string: the last box; else the marked line; else the section's last line when it
    can be a conclusion. The checker's verdict is `judge`, which also accepts a right last line."""
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    lines = answer_lines(text, no_think)
    best = marked_line(lines)
    if best:
        return best
    return last_line_answer(text, no_think)


def judge(gold: str, text: str, no_think: bool = False):
    """The checker's verdict on a generation: (right, answer, rule). `answer` is the string that
    was compared with the gold, `rule` says which line it came from:
      "boxed":        a complete box exists, and only it is read (right or wrong)
      "marked line":  the last line carrying an answer marker (right, or the last line cannot be
                      a conclusion)
      "last line":    the section's last line, when it is right and the marked line is not, or
                      when no line carries a marker
      None:           nothing was read (no section, or only a line that cannot be a conclusion)"""
    boxed = last_boxed(text)
    if boxed is not None:
        return answers_match(gold, boxed), boxed, "boxed"
    marked = marked_line(answer_lines(text, no_think))
    last = last_line_answer(text, no_think)
    for rule, ans in (("marked line", marked), ("last line", last)):
        if ans is not None and answers_match(gold, ans):
            return True, ans, rule
    if marked is not None:
        return False, marked, "marked line"
    return False, last, ("last line" if last is not None else None)


def load_texts(name: str):
    """<name>_texts.json, or the gzipped copy when the plain file is absent."""
    here = Path(__file__).resolve().parent
    plain = here / f"{name}_texts.json"
    if plain.exists():
        return json.loads(plain.read_text(encoding="utf-8"))
    with gzip.open(here / f"{name}_texts.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """Replace a JSON file only once the new content is complete on disk. Non-ASCII characters
    stay escaped: older scripts read these files without naming an encoding."""
    path, tmp = Path(path), Path(str(path) + ".tmp")
    tmp.write_bytes(json.dumps(obj, indent=1).encode("utf-8"))   # the same bytes on every platform
    tmp.replace(path)


def rescore(name: str, no_think: bool = False):
    """Re-score <name>.json from its generations; returns the validity problems (ISSUES #20)."""
    here = Path(__file__).resolve().parent
    texts = load_texts(name)
    run = json.loads((here / f"{name}.json").read_text(encoding="utf-8"))
    rows = run["rows"]
    correct = answered = 0
    for r in rows:
        right, pred, rule = judge(r["gold"], texts[r["id"]], no_think)
        r["pred_rescored"], r["ok_rescored"], r["rule"] = pred, right, rule
        correct += right
        answered += pred is not None
    n = len(rows)
    problems, validity = check_run(rows, texts, ok_key="ok_rescored")
    summary = dict(name=f"{name}_rescored", n=n,
                   accuracy=round(correct / n * 100, 1),
                   answered=round(answered / n * 100, 1),
                   # a run made by the phase-8 runner records no cap-hit rate
                   cap_hit_rate=run["summary"].get("cap_hit_rate"),
                   mean_tokens=run["summary"]["mean_tokens"],
                   rescored_with=RESCORED_WITH,
                   no_think=no_think,      # an input to every verdict, so it is on record with them
                   validity=dict(validity, problems=problems))
    print(json.dumps(summary))
    write_json(here / f"{name}_rescored.json", dict(summary=summary, rows=rows))
    return problems


if __name__ == "__main__":
    no_think = "--no-think" in sys.argv
    failed_any = False
    for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
        failed = rescore(name, no_think)
        if failed:
            print(f"RUN VALIDITY FAILED for {name} (ISSUES #20); marked in {name}_rescored.json: "
                  + "; ".join(failed), file=sys.stderr)
            failed_any = True
    sys.exit(2 if failed_any else 0)
