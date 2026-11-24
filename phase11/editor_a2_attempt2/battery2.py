"""Battery v2: MATH-500-style questions with multi-form answer checks.
Usage: ~/wsp3c-venv/bin/python battery2.py <model_dir> [max_tokens]"""
import re
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
TAG = sys.argv[3] if len(sys.argv) > 3 else "b2"
OUT = f"/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/battery2_{TAG}.txt"

# (question, [accepted answer forms])
QA = [
    ("What is $1+1$?", ["2"]),
    ("Compute $2 \\cdot 3 + 4$.", ["10"]),
    ("If $f(x)=3x+2$, what is $f(2)$?", ["8"]),
    ("What is $\\frac{1}{2} + \\frac{1}{4}$? Express your answer as a common fraction.",
     ["\\frac{3}{4}", "3/4", "3 / 4"]),
    ("Simplify $\\sqrt{50}$.", ["5\\sqrt{2}", "5\\sqrt2", "5 sqrt(2)"]),
    ("What is $15\\%$ of $80$?", ["12"]),
    ("Find the value of $x$ if $3x - 7 = 11$.", ["6"]),
    ("What is $7^2 - 3^2$?", ["40"]),
    ("What is the greatest common divisor of 24 and 36?", ["12"]),
    ("Convert $\\frac{7}{4}$ to a decimal.", ["1.75"]),
    ("What is $(-2)^3$?", ["-8", "- 8"]),
    ("A rectangle has length 6 and width 4. What is its area?", ["24"]),
    ("What is $2^{10}$?", ["1024"]),
    ("What is the smallest prime number?", ["2"]),
    ("How many sides does a hexagon have?", ["6"]),
    ("What is $\\frac{3}{5}$ expressed as a percentage?", ["60", "60\\%"]),
    ("If $3 + x = 10$, find $x$.", ["7"]),
    ("What is $100 - 37$?", ["63"]),
    ("Solve for $y$: $5y = 45$.", ["9"]),
    ("What is the area of a square with side length 5?", ["25"]),
]


def check(text, forms):
    for a in forms:
        pat = re.compile(r"(?<![\d.-])" + re.escape(a).replace("\\\\", "\\")
                         + r"(?![\d.])")
        if pat.search(text):
            return True
    return False


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
    s_tok = g_tok = 0
    s_stop = g_stop = 0
    for (q, forms), os_, og in zip(QA, outs_s, outs_g):
        ts, tg = os_.outputs[0].text, og.outputs[0].text
        cs, cg = check(ts, forms), check(tg, forms)
        s_ok += cs
        g_ok += cg
        s_tok += len(os_.outputs[0].token_ids)
        g_tok += len(og.outputs[0].token_ids)
        s_stop += os_.outputs[0].finish_reason == "stop"
        g_stop += og.outputs[0].finish_reason == "stop"
        lines.append(f"Q: {q}  [want {'/'.join(forms)}]")
        lines.append(f"  sampled({'OK' if cs else 'X'},{len(os_.outputs[0].token_ids)}tok,"
                     f"{os_.outputs[0].finish_reason}): {ts[:150].replace(chr(10), ' | ')!r}")
        lines.append(f"  greedy  ({'OK' if cg else 'X'},{len(og.outputs[0].token_ids)}tok,"
                     f"{og.outputs[0].finish_reason}): {tg[:150].replace(chr(10), ' | ')!r}")
    lines.append(f"SCORE2 sampled={s_ok}/{len(QA)} greedy={g_ok}/{len(QA)} | "
                 f"mean_toks s={s_tok/len(QA):.0f} g={g_tok/len(QA):.0f} | "
                 f"stops s={s_stop} g={g_stop}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print(lines[-1])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
