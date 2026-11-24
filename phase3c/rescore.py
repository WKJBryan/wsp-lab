r"""Phase-3c rescoring v2: format-aware extraction (ISSUES #10), fixed after audit.

Extraction order (identical for both arms, from saved generations):
  1. last \\boxed{...} anywhere;
  2. else, in the post-</think> section: the LAST line carrying an answer marker
     ("answer:", "final answer", "answer is", "value is", "coordinates are"),
     sliced at the marker (length-preserving view so indices are valid);
     empty payload -> next non-empty line;
  3. else the section's last line.
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
RESCORED_WITH = f"phase3c/rescore.py+phase3b/scorer.py#{CHECKSUM} (ISSUES #19)"

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
    seg = seg.strip().strip("*").strip().rstrip(".").strip()
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


def extract_answer(text: str, no_think: bool = False):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1]
    if no_think:  # phase 8: thinking-off output — the whole output is the answer section
        answer = text
    elif "</think>" not in text:
        return None
    else:
        answer = text.split("</think>", 1)[1]
    # a stray </think> token inside the answer section is not answer text (defect c)
    lines = [l.strip().replace("</think>", " ").strip() for l in answer.splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return None
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
    if best:
        return best
    return clean(lines[-1])


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
        pred = extract_answer(texts[r["id"]], no_think)
        r["pred_rescored"] = pred
        r["ok_rescored"] = answers_match(r["gold"], pred)
        correct += r["ok_rescored"]
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
