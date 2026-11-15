import json, sys

def main():

    from vllm import LLM, SamplingParams

    probs = json.load(open("practice.json"))
    want = {"gsm5570","gsm4131","gsm1398"}
    sel = [p for p in probs if p["id"] in want][:3] + probs[10:13]
    THINK_END = 248069
    llm = LLM(model="model", max_model_len=12000, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=1200, logprobs=20)
    msgs = [[{"role":"user","content":p["q"]}] for p in sel]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})
    summary = []
    for p,o in zip(sel,outs):
        text = o.outputs[0].text
        lps = o.outputs[0].logprobs  # list of {id: Logprob}
        track = []
        for i,d in enumerate(lps):
            arg = max(d.values(), key=lambda L: L.logprob)
            te = d.get(THINK_END)
            track.append((i, arg.decoded_token, round(arg.logprob,2),
                          (round(te.logprob,2), te.rank) if te else (None,">20")))
        # positions where the loop signature occurs
        looppos = [i for i,t in enumerate(track) if t[1] is not None and "Wait" in t[1]]
        best = sorted(track, key=lambda t: -(t[3][0] if t[3][0] is not None else -99))[:5]
        summary.append(dict(id=p["id"], ntok=len(track), closed="</think>" in text,
            n_wait=len(looppos),
            wait_examples=[track[i] for i in looppos[:4]],
            best_te_positions=best,
            tail=text[-150:]))
    print(json.dumps(summary, indent=1, default=str))

if __name__ == "__main__":
    main()
