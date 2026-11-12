"""Phase-3c rescoring v2: format-aware extraction (ISSUES #10), fixed after audit.

Extraction order (identical for both arms, from saved generations):
  1. last \\boxed{...} anywhere;
  2. else, in the post-</think> section: the LAST line carrying an answer marker
     ("answer:", "final answer", "answer is", "value is", "coordinates are"),
     sliced at the marker (length-preserving view so indices are valid);
     empty payload -> next non-empty line;
  3. else the section's last line.
Payload cleanup: \\text{X}->X, strip $..$, \\[ \\], bold, trailing period,
take the part after the last '='; then prefer the last $math$ span, else the
last number-ish token. Truncated thinking => no answer => wrong.

Usage: uv run python phase3c/rescore.py <run_name>
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase3b"))
from scorer import answers_match, canonical  # noqa: E402

MARKER_RE = re.compile(
    r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should",
    re.IGNORECASE,
)
STAR_VIEW = str.maketrans({"*": " "})


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
    seg = seg.strip().strip("$").strip()
    if not seg:
        return None
    maths = re.findall(r"\$([^$]+)\$", seg)
    if maths:
        return canonical(maths[-1])
    if "\\" in seg:  # LaTeX payload — trust it whole
        return canonical(seg)
    if re.search("[a-zA-Z]", seg):  # prose remainder — last number, else last word
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
        if nums:
            return nums[-1]
        words = re.findall(r"[A-Za-z]+", seg)
        return canonical(words[-1]) if words else None
    return canonical(seg)


def extract_answer(text: str):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1]
    if "</think>" not in text:
        return None
    answer = text.split("</think>", 1)[1]
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
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


def rescore(name: str):
    here = Path(__file__).resolve().parent
    texts = json.loads((here / f"{name}_texts.json").read_text())
    run = json.loads((here / f"{name}.json").read_text())
    rows = run["rows"]
    correct = answered = 0
    for r in rows:
        pred = extract_answer(texts[r["id"]])
        r["pred_rescored"] = pred
        r["ok_rescored"] = answers_match(r["gold"], pred)
        correct += r["ok_rescored"]
        answered += pred is not None
    n = len(rows)
    summary = dict(name=f"{name}_rescored", n=n,
                   accuracy=round(correct / n * 100, 1),
                   answered=round(answered / n * 100, 1),
                   cap_hit_rate=run["summary"]["cap_hit_rate"],
                   mean_tokens=run["summary"]["mean_tokens"])
    print(json.dumps(summary))
    (here / f"{name}_rescored.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    rescore(sys.argv[1])
