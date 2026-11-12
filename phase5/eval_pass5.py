"""Phase-5 practice evaluation (WSL, vLLM): 64 GSM8K problems, cap 8192, WITH accuracy.

Usage: $HOME/wsp3c-venv/bin/python eval_pass5.py <model_dir> <practice.json> <pass_index>
Prints one JSON line: accuracy, termination stats, sample tails.
"""

import json
import re
import sys
import time

from vllm import LLM, SamplingParams

sys.path.insert(0, "/home/bladless/wsp3c")
from scorer import answers_match  # noqa: E402

CAP = 8192
SEED = 20261105

MARKER_RE = re.compile(
    r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should",
    re.IGNORECASE,
)
STAR_VIEW = str.maketrans({"*": " "})


def extract_answer(text):
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
        lv = l.lower().translate(STAR_VIEW)
        m = MARKER_RE.search(lv)
        if m:
            payload = lv[m.end():].lstrip(":").strip()
            if not payload and i + 1 < len(lines):
                payload = lines[i + 1].lower().translate(STAR_VIEW).lstrip(":").strip()
            c = clean(payload)
            if c:
                best = c
    return best or clean(lines[-1])


def clean(seg):
    seg = seg.strip().rstrip(".")
    bolds = re.findall(r"\*{1,2}([^*]+)\*{1,2}", seg)
    if bolds:
        seg = bolds[-1]
    seg = seg.strip().strip("$").strip()
    if not seg:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
    if re.search("[a-zA-Z]", seg) and nums:
        return nums[-1]
    return seg or None


def main():
    model_dir, problems_path, pass_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
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

    n, correct, capped, closed = len(problems), 0, 0, 0
    samples = []
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        capped += o.outputs[0].finish_reason == "length"
        closed += "</think>" in text
        pred = extract_answer(text)
        correct += answers_match(p["gold"], pred)
        if o.outputs[0].finish_reason == "length" and len(samples) < 3:
            samples.append(dict(id=p["id"], tail=text[-200:]))
    print(json.dumps(dict(
        n=n, accuracy=round(correct / n * 100, 1), cap_hit_rate=round(capped / n * 100, 1),
        closed_rate=round(closed / n * 100, 1),
        mean_tokens=round(sum(len(o.outputs[0].token_ids) for o in outs) / n, 0),
        wall_s=round(time.time() - t0, 1), stuck_samples=samples,
    )))


if __name__ == "__main__":
    main()
