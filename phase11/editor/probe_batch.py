"""Free probe battery: several questions in one GPU load (adapted from probe_template).

Usage (in WSL): ~/wsp3c-venv/bin/python probe_batch.py [max_tokens] [tag]
Writes full generations to probe_out_<tag>.txt in this dir.
"""
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"

QUESTIONS = [
    "What is 17 * 23? Show your work briefly.",
    "Find the area of a triangle with side lengths 3, 4, and 5.",
    "Compute 2^10 - 3^5.",
]


def main():
    mt = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    tag = sys.argv[2] if len(sys.argv) > 2 else "a"

    llm = LLM(model=MODEL, max_model_len=12000, gpu_memory_utilization=0.85,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=mt, seed=7)
    chats = [[{"role": "user", "content": q}] for q in QUESTIONS]
    outs = llm.chat(chats, sp)
    lines = []
    for q, o in zip(QUESTIONS, outs):
        out = o.outputs[0]
        lines.append(f"=== Q: {q}\nfinish: {out.finish_reason} | tokens: {len(out.token_ids)}")
        lines.append(out.text)
        lines.append("=== END\n")
    txt = "\n".join(lines)
    dest = Path(__file__).resolve().parent / f"probe_out_{tag}.txt"
    dest.write_text(txt, encoding="utf-8")
    print(txt)
    print("[saved]", dest)


if __name__ == "__main__":
    main()
