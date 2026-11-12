"""Phase-5 probe: n=4 vs n=1 soup hypothesis (2 problems, cap 32768, dose175)."""

import json

from vllm import LLM, SamplingParams


def main():
    problems = json.loads(open("/home/bladless/wsp5/problems500.json").read())[:2]
    llm = LLM(model="/home/bladless/wsp5/dose175", max_model_len=40000, gpu_memory_utilization=0.88)

    def run(sp, label):
        msgs = [[{"role": "user", "content": p["q"]}] for p in problems]
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})
        for j, o in enumerate(outs):
            for k, c in enumerate(o.outputs):
                t = c.text
                close = "</think>" in t
                print(f"{label} p{j}#{k}: tok={len(c.token_ids)} finish={c.finish_reason} "
                      f"closed={close} tail={t[-60:]!r}", flush=True)

    base = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=1.5, max_tokens=32768)
    run(SamplingParams(**base, seed=20261105, n=4), "N4")
    for k in range(2):
        run(SamplingParams(**base, seed=20261106 + k, n=1), f"N1-{k}")
    print("PROBE DONE")


if __name__ == "__main__":
    main()
