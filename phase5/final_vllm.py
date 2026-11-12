"""Phase-5 finals (WSL, vLLM): full MATH-500, n=4 samples per problem, cap 32768.

Same decoding as phase 3c; per-sample scoring with the phase-5 extractor;
score = mean accuracy across all 2,000 generations (matches the original's
"averaged over 4 samples"). Also reports mean-over-samples per problem.

Usage: $HOME/wsp3c-venv/bin/python final_vllm.py <model_dir> <name>
Writes <name>.json + <name>_texts.json in cwd.
"""

import gzip
import json
import re
import sys
import time

from vllm import LLM, SamplingParams

sys.path.insert(0, "/home/bladless/wsp3c")
from scorer import answers_match  # noqa: E402

CAP = 32768
SEED = 20261105
NS = 4

MARKER_RE = re.compile(
    r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should",
    re.IGNORECASE,
)
STAR_VIEW = str.maketrans({"*": " "})


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


def main():
    model_dir, name = sys.argv[1], sys.argv[2]
    problems = json.loads(open("/home/bladless/wsp5/problems500.json").read())

    llm = LLM(model=model_dir, max_model_len=40000, gpu_memory_utilization=0.92)
    sp = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        max_tokens=CAP, seed=SEED, n=NS,
    )
    msgs = [[{"role": "user", "content": p["q"]}] for p in problems]
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})

    rows, texts = [], {}
    total_ok = total_gen = total_capped = 0
    per_problem = []
    for p, o in zip(problems, outs):
        oks, text_list = [], []
        for comp in o.outputs:
            text = comp.text
            pred = extract_answer(text)
            ok = answers_match(p["gold"], pred)
            oks.append(ok)
            total_ok += ok
            total_gen += 1
            total_capped += comp.finish_reason == "length"
            text_list.append(text)
            texts[f"{p['id']}#{len(text_list)-1}"] = text
        per_problem.append(sum(oks) / len(oks))
        rows.append(dict(id=p["id"], gold=p["gold"], sample_ok=oks,
                         mean_ok=round(sum(oks) / len(oks), 3)))

    summary = dict(name=name, model=model_dir, cap=CAP, n=len(problems), samples=NS,
                   accuracy_mean_over_generations=round(total_ok / total_gen * 100, 1),
                   accuracy_mean_over_problems=round(sum(per_problem) / len(per_problem) * 100, 1),
                   cap_hit_rate=round(total_capped / total_gen * 100, 1),
                   wall_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    json.dump(dict(summary=summary, rows=rows), open(f"{name}.json", "w"), indent=1)
    with gzip.open(f"{name}_texts.json.gz", "wt") as f:
        json.dump(texts, f)


if __name__ == "__main__":
    main()
