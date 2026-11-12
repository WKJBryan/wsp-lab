"""Phase-5 finals v2 (WSL): 4 sequential n=1 passes of 500 problems (attempt #1's
2000-in-flight run produced token soup; probe cleared n=4/depth — concurrency suspect).

Usage: $HOME/wsp3c-venv/bin/python final_vllm2.py <model_dir> <name>
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
PASSES = 4

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
    msgs = [[{"role": "user", "content": p["q"]}] for p in problems]

    llm = LLM(model=model_dir, max_model_len=40000, gpu_memory_utilization=0.88,
              enable_prefix_caching=False)

    rows = {p["id"]: dict(gold=p["gold"], sample_ok=[]) for p in problems}
    texts = {}
    total_ok = total_gen = total_capped = 0
    t0 = time.time()
    for s in range(PASSES):
        sp = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
            presence_penalty=1.5, repetition_penalty=1.0,
            max_tokens=CAP, seed=SEED + s, n=1,
        )
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})
        p_ok = p_cap = 0
        for p, o in zip(problems, outs):
            c = o.outputs[0]
            t = c.text
            ok = answers_match(p["gold"], extract_answer(t))
            rows[p["id"]]["sample_ok"].append(ok)
            total_ok += ok
            total_gen += 1
            cap = c.finish_reason == "length"
            total_capped += cap
            p_ok += ok
            p_cap += cap
            texts[f"{p['id']}#s{s}"] = t
        print(f"  pass {s}: acc {p_ok}/{len(problems)} cap-hits {p_cap}", flush=True)
        # in-flight soup check: tail of one long generation
        for p, o in zip(problems, outs):
            if len(o.outputs[0].token_ids) > 15000:
                print(f"  sanity tail: {o.outputs[0].text[-70:]!r}", flush=True)
                break

    per_problem_mean = [sum(r["sample_ok"]) / len(r["sample_ok"]) for r in rows.values()]
    summary = dict(name=name, model=model_dir, cap=CAP, n=len(problems), samples=PASSES,
                   accuracy_mean_over_generations=round(total_ok / total_gen * 100, 1),
                   accuracy_mean_over_problems=round(sum(per_problem_mean) / len(per_problem_mean) * 100, 1),
                   cap_hit_rate=round(total_capped / total_gen * 100, 1),
                   wall_s=round(time.time() - t0, 1))
    print(json.dumps(summary))
    json.dump(dict(summary=summary, rows=list(rows.values())), open(f"{name}.json", "w"), indent=1)
    with gzip.open(f"{name}_texts.json.gz", "wt") as f:
        json.dump(texts, f)


if __name__ == "__main__":
    main()
