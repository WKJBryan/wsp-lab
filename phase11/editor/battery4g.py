"""Battery v4: 120+ questions (programmatic + curated), strict lastnum scoring.
Usage: ~/wsp3c-venv/bin/python battery4.py <model_dir> [max_tokens] [tag]"""
import re
import sys

from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
TAG = sys.argv[3] if len(sys.argv) > 3 else "b4"
OUT = f"/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/battery4g_{TAG}.txt"


def build_qa():
    qa = []
    # programmatic arithmetic (60)
    import random
    rng = random.Random(11)
    for _ in range(12):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        qa.append((f"What is ${a} + {b}$?", str(a + b)))
    for _ in range(12):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        qa.append((f"What is ${a} + {b}$?", str(a + b)))
    for _ in range(12):
        a, b = rng.randint(5, 99), rng.randint(2, min(a, 50))
        qa.append((f"What is ${a} - {b}$?", str(a - b)))
    for _ in range(12):
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        qa.append((f"What is ${a} \\times {b}$?", str(a * b)))
    for _ in range(6):
        b = rng.randint(2, 12)
        c = b * rng.randint(2, 12)
        qa.append((f"What is ${c} \\div {b}$?", str(c // b)))
    for _ in range(6):
        p = rng.choice([10, 20, 25, 50, 75])
        n = rng.choice([20, 40, 60, 80, 100, 200])
        qa.append((f"What is ${p}\\%$ of ${n}$?", str(int(p * n / 100))))
    # curated facts / retrieval (30)
    facts = [
        ("What is the smallest prime number?", "2"),
        ("What is the smallest composite number?", "4"),
        ("How many sides does a triangle have?", "3"),
        ("How many sides does a hexagon have?", "6"),
        ("How many sides does an octagon have?", "8"),
        ("How many degrees are in a right angle?", "90"),
        ("How many degrees are in a straight angle?", "180"),
        ("How many degrees are in a full circle?", "360"),
        ("How many minutes are in an hour?", "60"),
        ("How many hours are in a day?", "24"),
        ("How many days are in a week?", "7"),
        ("How many months are in a year?", "12"),
        ("What is the value of $x$ if $x = 5$?", "5"),
        ("What is $2 + 2$?", "4"),
        ("What is $10 - 10$?", "0"),
        ("What is $1 \\times 1$?", "1"),
        ("How many corners does a square have?", "4"),
        ("How many players are on a soccer team on the field?", "11"),
        ("What is the atomic number of hydrogen?", "1"),
        ("How many colors are in a rainbow?", "7"),
        ("What is $5 \\times 5$?", "25"),
        ("What is $6 \\times 7$?", "42"),
        ("What is $100 + 100$?", "200"),
        ("What is $1000 - 1$?", "999"),
        ("A dozen equals how many?", "12"),
        ("How many wheels does a tricycle have?", "3"),
        ("How many strings does a standard guitar have?", "6"),
        ("What is the largest single-digit number?", "9"),
        ("What is the smallest positive even number?", "2"),
        ("How many letters are in the English alphabet?", "26"),
    ]
    qa.extend(facts)
    # simple algebra (30)
    for i in range(15):
        a = rng.randint(2, 9)
        x = rng.randint(2, 20)
        b = rng.randint(1, 50)
        qa.append((f"Solve for $x$: ${a}x = {a * x}$.", str(x)))
    for i in range(15):
        x = rng.randint(2, 30)
        b = rng.randint(1, 40)
        qa.append((f"If $x + {b} = {x + b}$, what is $x$?", str(x)))
    return qa


def norm(s):
    s = s.strip().rstrip(".").strip("$ ")
    s = s.replace("\\!", "").replace("\\,", "").replace(" ", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"^\((.*)\)$", r"\1", s)
    return s


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def main():
    QA = build_qa()
    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.55,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=0.0,
                        max_tokens=MAXTOK, seed=7)
    msgs = [[{"role": "user", "content": q}] for q, _ in QA]
    outs = llm.chat(msgs, sp)
    lines = []
    ok = 0
    stops = 0
    toks = 0
    first200_correct_stop = 0
    for (q, a), o in zip(QA, outs):
        t = o.outputs[0].text
        toks += len(o.outputs[0].token_ids)
        stops += o.outputs[0].finish_reason == "stop"
        ln = last_number(t)
        good = ln is not None and norm(ln) == norm(a)
        ok += good
        # first-attempt check: number within first 200 chars
        m = re.findall(r"-?\d+(?:\.\d+)?", t[:200])
        first200_correct_stop += bool(m and norm(m[-1]) == norm(a))
        lines.append(f"Q: {q} [want {a}] lastnum={ln!r} {'OK' if good else ''}")
    n = len(QA)
    lines.append(f"SCORE4 lastnum={ok}/{n} first200={first200_correct_stop}/{n} "
                 f"mean_tok={toks/n:.0f} stops={stops}/{n}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print(lines[-1])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
