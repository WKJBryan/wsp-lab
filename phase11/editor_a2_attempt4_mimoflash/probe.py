"""Free probe: generate from the sandbox model and show raw behavior.

Usage:
  ~/wsp3c-venv/bin/python probe.py "q1" "q2" ... [--max-tokens 512] [--greedy]

Loads the engine once and runs all questions. Never run during an official pass.
"""

import argparse

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    args = ap.parse_args()

    llm = LLM(model=MODEL, max_model_len=12000, gpu_memory_utilization=args.gpu_mem,
              moe_backend="triton", enforce_eager=True, max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})

    if args.greedy:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    else:
        sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64,
                            max_tokens=args.max_tokens, seed=args.seed)

    chats = [[{"role": "user", "content": q}] for q in args.questions]
    outs = llm.chat(chats, sp)
    for q, out in zip(args.questions, outs):
        o = out.outputs[0]
        print("=" * 60)
        print("Q:", q)
        print("finish:", o.finish_reason, "| tokens:", len(o.token_ids))
        print("--- generation ---")
        print(o.text)
        print("--- end ---")


if __name__ == "__main__":
    main()
