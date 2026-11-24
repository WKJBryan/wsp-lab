"""Probe battery: fixed math questions, eval-matched decoding + greedy, correctness check.
Usage: ~/wsp3c-venv/bin/python battery.py <model_dir> [max_tokens]
"""
import json
import re
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
OUT = "/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/battery_out.txt"

# (question, expected numeric answer)
QA = [
    ("What is 17 * 23? Answer with just the number.", "391"),
    ("What is the greatest common divisor of 48 and 64? Answer with just the number.", "16"),
    ("What is 2 + 3? Answer with just the number.", "5"),
    ("What is 12 * 12? Answer with just the number.", "144"),
    ("What is 100 / 4? Answer with just the number.", "25"),
    ("What is 7 * 8? Answer with just the number.", "56"),
    ("What is the square root of 144? Answer with just the number.", "12"),
    ("What is 15% of 200? Answer with just the number.", "30"),
    ("What is 3^4? Answer with just the number.", "81"),
    ("What is -5 + 12? Answer with just the number.", "7"),
    ("Solve for x: 2x + 5 = 15. Answer with just the value of x.", "5"),
    ("What is 9 * 6? Answer with just the number.", "54"),
]


def check(text, ans):
    # last standalone occurrence of the expected number
    pat = re.compile(r"(?<![\d.])" + re.escape(ans) + r"(?![\d.])")
    return bool(pat.search(text))


def main():
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.55,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp_sample = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                               max_tokens=MAXTOK, seed=7)
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=MAXTOK)
    msgs = [[{"role": "user", "content": q}] for q, _ in QA]
    outs_s = llm.chat(msgs, sp_sample)
    outs_g = llm.chat(msgs, sp_greedy)

    lines = []
    s_ok = g_ok = 0
    for (q, a), os_, og in zip(QA, outs_s, outs_g):
        ts, tg = os_.outputs[0].text, og.outputs[0].text
        cs, cg = check(ts, a), check(tg, a)
        s_ok += cs
        g_ok += cg
        lines.append(f"Q: {q}  [expect {a}]")
        lines.append(f"  sampled({'OK' if cs else 'X'}, {len(os_.outputs[0].token_ids)}tok, "
                     f"{os_.outputs[0].finish_reason}): {ts[:180].replace(chr(10), ' | ')!r}")
        lines.append(f"  greedy  ({'OK' if cg else 'X'}, {len(og.outputs[0].token_ids)}tok, "
                     f"{og.outputs[0].finish_reason}): {tg[:180].replace(chr(10), ' | ')!r}")
    lines.append(f"SCORE sampled={s_ok}/{len(QA)} greedy={g_ok}/{len(QA)}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[-1:]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
