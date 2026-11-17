"""Free probe: a few math questions to observe failure mode."""
from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"

qs = [
    "What is 17 * 23? Show your work briefly.",
    "Compute the derivative of x^3 + 2x with respect to x.",
    "What is the greatest common divisor of 48 and 64? Answer with just the number.",
]


def main():
    llm = LLM(model=MODEL, max_model_len=6144, gpu_memory_utilization=0.55,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=4096, seed=7)
    outs = llm.chat([[{"role": "user", "content": q}] for q in qs], sp)
    for q, o in zip(qs, outs):
        print("=" * 70)
        print("Q:", q)
        print("finish:", o.outputs[0].finish_reason, "| tokens:", len(o.outputs[0].token_ids))
        print("--- generation ---")
        print(o.outputs[0].text)


if __name__ == "__main__":
    main()
