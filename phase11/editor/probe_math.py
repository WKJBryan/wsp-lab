"""Probe: several prompts, two decoding configs, observe failure mode."""
from vllm import LLM, SamplingParams

MODEL = "/home/user/wsp11/sandbox/model"

QS = [
    "What is 17 * 23? Show your work briefly.",
    "Solve for x: 3x + 5 = 20.",
    "What is the capital of France? Answer in one word.",
    "Compute the derivative of f(x) = x^3 + 2x.",
    "A rectangle has width 4 and height 7. What is its area?",
]

def main():
    llm = LLM(model=MODEL, max_model_len=12000, gpu_memory_utilization=0.60,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})

    configs = {
        "greedy": SamplingParams(temperature=0.0, max_tokens=1024),
        "sample7": SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=1024, seed=7),
    }

    for cname, sp in configs.items():
        print(f"\n########## CONFIG {cname} ##########", flush=True)
        msgs = [[{"role": "user", "content": q}] for q in QS]
        outs = llm.chat(msgs, sp)
        for q, o in zip(QS, outs):
            text = o.outputs[0].text
            print(f"\n=== Q: {q}")
            print(f"finish: {o.outputs[0].finish_reason} | tokens: {len(o.outputs[0].token_ids)}")
            print(text[:2000])
            print("---END---", flush=True)

if __name__ == "__main__":
    main()
