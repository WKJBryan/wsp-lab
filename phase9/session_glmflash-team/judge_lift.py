import json, random, gc
import torch

BASE = "/home/bladless/wsp4/sandbox/model"
PATCH = "/home/bladless/wsp4/sandbox/dose_u_a0.2"
TID = 248069

def main():
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    ps = json.load(open("/home/bladless/wsp4/sandbox/practice.json"))
    replays = json.load(open("/home/bladless/wsp4/sandbox/probe_b_full.json"))

    # position plan per looper: stride + near-miss surges + tail of natural-closers
    plan = []
    for r in replays:
        toks, steps = r["tokens"], r["steps"]
        pos = set(range(0, len(toks), 75))
        surge = []
        for i, s in enumerate(steps):
            top = s.get("top", {})
            if str(TID) in top and top[str(TID)] >= -12.5:
                surge.append((i, top[str(TID)]))
        pos |= {i for i, _ in surge}
        if r.get("finish") == "stop":
            pos |= set(range(max(0, len(toks)-30), len(toks)))
        pos = sorted(x for x in pos if 0 < x <= len(toks))
        plan.append({"id": r["id"], "toks": toks, "pos": pos,
                     "surge": dict(surge), "finish": r["finish"]})

    prompts_by_id = {}
    for p in ps:
        if p["id"] in {x["id"] for x in plan}:
            ids = tok.apply_chat_template([{"role": "user", "content": p["q"]}],
                                          add_generation_prompt=True,
                                          enable_thinking=True, tokenize=True)
            if hasattr(ids, "input_ids"): ids = ids.input_ids
            if ids and isinstance(ids[0], list): ids = ids[0]
            prompts_by_id[p["id"]] = list(map(int, ids))

    rows = []
    for x in plan:
        for i in x["pos"]:
            ids = prompts_by_id[x["id"]] + x["toks"][:i] + [TID]
            rows.append({"id": x["id"], "i": i, "surge_lp": x["surge"].get(i),
                         "ids": ids})
    print("total probe positions:", len(rows))

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def run(model_dir, label):
        llm = LLM(model=model_dir, max_model_len=12000,
                  gpu_memory_utilization=0.30, enforce_eager=True)
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rows], sp)
        got = []
        for r, o in zip(rows, outs):
            plp = o.prompt_logprobs
            d = plp[-1]  # logprob of the forced 248069
            lp = d[TID].logprob if (d and TID in d) else float("nan")
            got.append({"id": r["id"], "i": r["i"], "surge_lp": r["surge_lp"],
                        "lp": lp})
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return got

    base = run(BASE, "base")
    patch = run(PATCH, "patch")
    bmap = {(g["id"], g["i"]): g["lp"] for g in base}
    out = []
    for g in patch:
        lb = bmap[(g["id"], g["i"])]
        out.append({"id": g["id"], "i": g["i"], "lp_base": lb, "lp_patch": g["lp"],
                    "lift": g["lp"] - lb, "surge_lp": g["surge_lp"]})
    json.dump(out, open("/home/bladless/wsp4/sandbox/judge_lift.json", "w"), indent=0)

    import statistics as st
    lifts = [o["lift"] for o in out]
    qs = torch.tensor(lifts).quantile(torch.tensor([.1,.25,.5,.75,.9,.95,.99])).tolist()
    print("LIFT nats: n=%d  p10=%.2f p25=%.2f med=%.2f p75=%.2f p90=%.2f p95=%.2f p99=%.2f  min=%.2f max=%.2f"
          % (len(lifts), qs[0], qs[1], qs[2], qs[3], qs[4], qs[5], qs[6], min(lifts), max(lifts)))
    near = [o for o in out if o["surge_lp"] is not None]
    print("surge-zone n=%d med lift=%.2f p90=%.2f" % (len(near),
          st.median([o["lift"] for o in near]),
          torch.tensor([o["lift"] for o in near]).quantile(0.9).item()))
    deep = [o for o in out if o["surge_lp"] is None]
    print("deep-zone  n=%d med lift=%.2f p90=%.2f" % (len(deep),
          st.median([o["lift"] for o in deep]),
          torch.tensor([o["lift"] for o in deep]).quantile(0.9).item()))
    # post-lift fireability: would patched lp beat top-1 at these positions?
    # baseline top1 lp not stored for all; use surge cross-check + absolute patched lp
    for o in out:
        if o["surge_lp"] is not None and abs(o["surge_lp"] - o["lp_base"]) > 0.15:
            print("CROSSCHECK-MISMATCH", o["id"], o["i"], "stored", o["surge_lp"], "forced", o["lp_base"])
    worst = sorted(out, key=lambda o: -o["lift"])[:10]
    print("top lifts:", [(o["id"], o["i"], round(o["lift"],2), round(o["lp_patch"],2)) for o in worst])
    # how many positions now have patch lp > -2.5 (roughly top-20/fireable under presence penalty)?
    hot = [o for o in out if o["lp_patch"] > -2.5]
    print("positions with patch lp > -2.5:", len(hot), [(o["id"], o["i"], round(o["lp_patch"],2)) for o in hot[:12]])

if __name__ == "__main__":
    main()
