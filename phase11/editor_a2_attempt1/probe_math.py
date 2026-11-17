"""Probe the sandbox model with MATH-500-style problems (free, uncounted).

Usage: probe_math.py [greedy|fixed]
  fixed  = eval-like decoding from probe template (temp 1.0, top_p 0.95, top_k 64, seed 7)
  greedy = temperature 0 for clean diagnosis
"""
import sys

from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"

QUESTIONS = [
    "What is the greatest common divisor of 84 and 144?",
    "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
    "If $f(x)=\\dfrac{x+4}{x^2+ax+b}$, and $f(x)$ has two vertical asymptotes at $x=1$ and $x=-2$, find the sum of $a$ and $b$.",
    "A parabola with equation $y=x^2+bx+c$ passes through the points $(2,3)$ and $(4,3)$. What is $c$?",
    "Compute $17\\times 23 - 11$.",
    "Simplify $\\sqrt[3]{72}+\\sqrt[3]{375}$.",
    "What is $1+2+3+\\cdots+100$?",
    "Let $\\mathbf{A}$ be the matrix $\\begin{pmatrix}2 & 1\\\\1 & 2\\end{pmatrix}$. Find $\\det \\mathbf{A}$.",
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fixed"

    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=0.60,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})

    for i, q in enumerate(QUESTIONS):
        if mode == "greedy":
            sp = SamplingParams(temperature=0.0, max_tokens=1024)
        else:
            sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=2048, seed=7)
        outs = llm.chat([[{"role": "user", "content": q}]], sp)
        text = outs[0].outputs[0].text
        print(f"\n{'='*80}\nQ{i+1}: {q}")
        print(f"finish: {outs[0].outputs[0].finish_reason} | tokens: {len(outs[0].outputs[0].token_ids)}")
        print("--- generation ---")
        print(text)


if __name__ == "__main__":
    main()
