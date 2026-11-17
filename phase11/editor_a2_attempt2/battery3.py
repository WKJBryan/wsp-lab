"""Battery v3: 40 MATH-style questions, STRICT scorer:
- boxed score: last \\boxed{...} content, normalized, must match expected
- lastnum score: last standalone number in text must match expected
Usage: ~/wsp3c-venv/bin/python battery3.py <model_dir> [max_tokens] [tag]"""
import re
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
TAG = sys.argv[3] if len(sys.argv) > 3 else "b3"
OUT = f"/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/battery3_{TAG}.txt"

# (question, expected answer string, accepted normalized variants)
QA = [
    ("What is $1+1$?", "2", ["2"]),
    ("What is $2+3$?", "5", ["5"]),
    ("Compute $2 \\cdot 3 + 4$.", "10", ["10"]),
    ("What is $12 \\times 12$?", "144", ["144"]),
    ("What is $100 - 37$?", "63", ["63"]),
    ("What is $7 \\cdot 8$?", "56", ["56"]),
    ("What is $2^{10}$?", "1024", ["1024"]),
    ("What is $3^4$?", "81", ["81"]),
    ("What is $(-2)^3$?", "-8", ["-8"]),
    ("What is $\\sqrt{144}$?", "12", ["12"]),
    ("What is $\\frac{1}{2} + \\frac{1}{4}$?", "\\frac{3}{4}", ["\\frac{3}{4}", "3/4", "0.75"]),
    ("What is $\\frac{3}{5}$ as a percentage?", "60", ["60", "60\\%"]),
    ("Convert $\\frac{7}{4}$ to a decimal.", "1.75", ["1.75"]),
    ("What is $15\\%$ of $80$?", "12", ["12"]),
    ("If $f(x)=3x+2$, what is $f(2)$?", "8", ["8"]),
    ("If $f(x)=x^2-1$, what is $f(3)$?", "8", ["8"]),
    ("Find $x$ if $3x - 7 = 11$.", "6", ["6"]),
    ("Solve for $y$: $5y = 45$.", "9", ["9"]),
    ("If $x = 7$, what is $x$?", "7", ["7"]),
    ("If $3 + x = 10$, find $x$.", "7", ["7"]),
    ("What is the greatest common divisor of 24 and 36?", "12", ["12"]),
    ("What is the least common multiple of 4 and 6?", "12", ["12"]),
    ("What is the smallest prime number?", "2", ["2"]),
    ("How many sides does a hexagon have?", "6", ["6"]),
    ("How many degrees are in a right angle?", "90", ["90"]),
    ("A rectangle has length 6 and width 4. What is its area?", "24", ["24"]),
    ("What is the area of a square with side length 5?", "25", ["25"]),
    ("What is the perimeter of a square with side length 3?", "12", ["12"]),
    ("Simplify $\\sqrt{50}$.", "5\\sqrt{2}", ["5\\sqrt{2}", "5\\sqrt2"]),
    ("Simplify $2x + 3x$.", "5x", ["5x"]),
    ("Combine like terms: $7y - 2y$.", "5y", ["5y"]),
    ("What is $x$ if $2x = x + 5$?", "5", ["5"]),
    ("Evaluate $\\frac{10}{4} \\cdot 2$.", "5", ["5"]),
    ("What is $9 - 4 + 1$?", "6", ["6"]),
    ("If a train travels 60 miles per hour for 2 hours, how many miles does it travel?", "120", ["120"]),
    ("What is $\\frac{6}{8}$ in simplest form?", "\\frac{3}{4}", ["\\frac{3}{4}", "3/4"]),
    ("What is the value of $\\pi$ rounded to two decimal places?", "3.14", ["3.14"]),
    ("What is $20 \\div 4$?", "5", ["5"]),
    ("What number is $50\\%$ of 14?", "7", ["7"]),
    ("If $A = \\{1, 2, 3\\}$, how many elements does $A$ have?", "3", ["3"]),
]


def norm(s):
    s = s.strip().rstrip(".").strip("$ ")
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"^\((.*)\)$", r"\1", s)
    return s


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace = text.find("{", idx)
    if brace == -1:
        return None
    depth = 0
    for j in range(brace, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:j]
    return text[brace + 1:]


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else None


def main():
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.55,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                        max_tokens=MAXTOK, seed=7)
    msgs = [[{"role": "user", "content": q}] for q, _, _ in QA]
    outs = llm.chat(msgs, sp)

    lines = []
    boxed_ok = lastnum_ok = boxed_any = 0
    toks = 0
    stops = 0
    for (q, a, variants), o in zip(QA, outs):
        t = o.outputs[0].text
        toks += len(o.outputs[0].token_ids)
        stops += o.outputs[0].finish_reason == "stop"
        bx = last_boxed(t)
        ln = last_number(t)
        acc = {norm(v) for v in [a] + variants}
        bok = bx is not None and norm(bx) in acc
        nok = ln is not None and norm(ln) in acc
        boxed_ok += bok
        lastnum_ok += nok
        boxed_any += bx is not None
        lines.append(f"Q: {q} [want {a}] boxed={bx!r} lastnum={ln!r} "
                     f"{'BOXOK' if bok else ''}{'NUMOK' if nok else ''}")
    lines.append(f"SCORE3 boxed={boxed_ok}/40 lastnum={lastnum_ok}/40 "
                 f"boxed_any={boxed_any}/40 mean_tok={toks/40:.0f} stops={stops}/40")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print(lines[-1])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
