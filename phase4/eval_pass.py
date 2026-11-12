"""Phase-4 harness-side practice evaluation (runs INSIDE WSL, from the sandbox).

64 fixed GSM8K-train problems, thinking on, temp 0.6 / top-p 0.95 / top-k 20 /
presence penalty 1.5, cap 8192, 1 sample. Prints one JSON line: summary + 3 sample tails.

Budget is HARNESS-ENFORCED: a counter file in the cwd allows at most 10 invocations;
each requires --hypothesis, which is appended to the ledger BEFORE the run.

Usage: $HOME/wsp3c-venv/bin/python eval_pass.py <model_dir> <problems.json> --hypothesis "..."
"""

import datetime
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

CAP = 8192
SEED = 20261105
MAX_PASSES = 10


def spend_pass(hypothesis: str) -> int:
    cnt_file = "eval_counter.txt"
    n = 0
    if os.path.exists(cnt_file):
        n = int(open(cnt_file).read().strip() or 0)
    if n >= MAX_PASSES:
        print(json.dumps(dict(error=f"budget exhausted: {n}/{MAX_PASSES} passes spent")))
        sys.exit(1)
    open(cnt_file, "w").write(str(n + 1))
    with open("ledger.md", "a") as f:
        f.write(
            f"| {n + 1} | {datetime.datetime.now().isoformat(timespec='seconds')} | "
            f"{hypothesis.replace('|', '/')} |\n"
        )
    return n + 1


def main():
    model_dir = sys.argv[1]
    problems_path = sys.argv[2]
    hypothesis = None
    if "--hypothesis" in sys.argv:
        hypothesis = sys.argv[sys.argv.index("--hypothesis") + 1]
    if not hypothesis:
        print(json.dumps(dict(error="--hypothesis is required (pre-logging rule)")))
        sys.exit(1)
    pass_index = spend_pass(hypothesis)
    problems = json.loads(open(problems_path).read())

    llm = LLM(model=model_dir, max_model_len=12000, gpu_memory_utilization=0.92)
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        max_tokens=CAP, seed=SEED + pass_index,
    )
    msgs = [[{"role": "user", "content": p["q"]}] for p in problems]
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})

    import re
    from fractions import Fraction

    # --- compact scorer (same logic family as phase3c/rescore.py + phase3b/scorer.py) ---
    MARKER_RE = re.compile(
        r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should", re.I)
    STAR_VIEW = str.maketrans({"*": " "})

    def canonical(s):
        s = s.strip().replace("\\!", "").replace("\\,", "").replace("\\ ", "")
        s = s.replace("\\left", "").replace("\\right", "").replace("$", "")
        s = s.replace("\\(", "").replace("\\)", "")
        s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
        s = s.replace(" ", "").rstrip(".")
        return s[1:] if s.startswith("+") else s

    def to_number(s):
        s = canonical(s)
        try:
            if re.fullmatch(r"-?\d+", s): return Fraction(int(s))
            if re.fullmatch(r"-?\d*\.\d+", s): return Fraction(s)
            m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
            if m and int(m.group(2)): return Fraction(int(m.group(1)), int(m.group(2)))
            m = re.fullmatch(r"\\frac\{(-?[^{}]+)\}\{(-?[^{}]+)\}", s)
            if m:
                a, b = to_number(m.group(1)), to_number(m.group(2))
                if a is not None and b is not None and b != 0: return a / b
        except Exception:
            return None
        return None

    def match(gold, pred):
        if pred is None: return False
        g, p = canonical(gold), canonical(pred)
        if g == p: return True
        gn, pn = to_number(g), to_number(p)
        return gn is not None and pn is not None and gn == pn

    def clean_inner(seg):
        seg = seg.strip().strip("$").strip()
        if not seg: return None
        maths = re.findall(r"\$([^$]+)\$", seg)
        if maths: return canonical(maths[-1])
        if "\\" in seg: return canonical(seg)
        if re.search("[a-zA-Z]", seg):
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
            if nums: return nums[-1]
            words = re.findall(r"[A-Za-z]+", seg)
            return canonical(words[-1]) if words else None
        return canonical(seg)

    def clean(seg):
        seg = seg.strip()
        seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
        seg = seg.replace("\\[", "").replace("\\]", "")
        if "=" in seg: seg = seg.split("=")[-1]
        seg = seg.strip().strip("*").strip().rstrip(".").strip()
        bolds = re.findall(r"\*\*([^*]+)\*\*", seg)
        return clean_inner(bolds[-1]) if bolds else clean_inner(seg)

    def extract(text):
        boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
        if boxed: return boxed[-1]
        if "</think>" not in text: return None
        answer = text.split("</think>", 1)[1]
        lines = [l.strip() for l in answer.splitlines() if l.strip()]
        if not lines: return None
        best = None
        for i, l in enumerate(lines):
            m = MARKER_RE.search(l.lower().translate(STAR_VIEW))
            if m:
                payload = l[m.end():].lstrip(":").strip()
                if not payload and i + 1 < len(lines):
                    payload = lines[i + 1]
                c = clean(payload)
                if c: best = c
        return best if best else clean(lines[-1])

    n, capped, closed, correct = len(problems), 0, 0, 0
    samples = []
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        finish = o.outputs[0].finish_reason
        n_tok = len(o.outputs[0].token_ids)
        capped += finish == "length"
        closed += "</think>" in text
        correct += match(p.get("gold", ""), extract(text))
        if finish == "length" and len(samples) < 3:
            samples.append(dict(id=p["id"], n_tokens=n_tok,
                                tail=text[-300:], head=text[:120]))
    print(json.dumps(dict(
        n=n, pass_index=pass_index,
        accuracy=round(correct / n * 100, 1),
        closed_rate=round(closed / n * 100, 1),
        cap_hit_rate=round(capped / n * 100, 1),
        closed_rate=round(closed / n * 100, 1),
        mean_tokens=round(sum(len(o.outputs[0].token_ids) for o in outs) / n, 0),
        wall_s=round(time.time() - t0, 1),
        note="practice accuracy intentionally NOT shown (blind protocol); termination stats only",
        samples=samples,
    )))


if __name__ == "__main__":
    main()
