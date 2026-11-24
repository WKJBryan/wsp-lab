"""Working offline probe template for the editor sandbox (free, uncounted).

Runs INSIDE WSL against the sandbox model. Mechanically correct flags for this
machine (triton MoE backend, multimodal inputs disabled, small VRAM slice so it
coexists with nothing else). Adapt freely — this is a starting point, not a rule.

Usage:
  ~/wsp3c-venv/bin/python probe_template.py "your question here"
"""

import sys

from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"

q = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23? Show your work briefly."

llm = LLM(model=MODEL, max_model_len=12000, gpu_memory_utilization=0.30,
          moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
          limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=2048, seed=7)
outs = llm.chat([[{"role": "user", "content": q}]], sp)
text = outs[0].outputs[0].text
print("finish:", outs[0].outputs[0].finish_reason, "| tokens:", len(outs[0].outputs[0].token_ids))
print("--- generation ---")
print(text)
