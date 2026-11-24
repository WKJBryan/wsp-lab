"""Free probe of the sandbox model (adapted from probe_template.py).

Usage (in WSL): ~/wsp3c-venv/bin/python probe.py "question" [max_tokens]
Prints finish reason, token count, and the generation.
"""
import sys

from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23? Show your work briefly."
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    llm = LLM(model=MODEL, max_model_len=12000, gpu_memory_utilization=0.85,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=mt, seed=7)
    outs = llm.chat([[{"role": "user", "content": q}]], sp)
    text = outs[0].outputs[0].text
    print("finish:", outs[0].outputs[0].finish_reason,
          "| tokens:", len(outs[0].outputs[0].token_ids))
    print("--- generation ---")
    print(text)


if __name__ == "__main__":
    main()
