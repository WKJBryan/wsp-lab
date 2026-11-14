import json, sys
from transformers import AutoTokenizer

MD = "/home/bladless/wsp4/sandbox/model"
tok = AutoTokenizer.from_pretrained(MD)
for s in ["<think>", "</think>", "<|im_end|>", "<|endoftext|>", "Wait", " no", ".", "\n", "$", "Wait,", " no."]:
    ids = tok.encode(s, add_special_tokens=False)
    print(repr(s), "->", ids)

p = json.load(open("/home/bladless/wsp4/sandbox/practice.json"))
print("n_problems", len(p), "keys", list(p[0].keys()))
keep = {}
for item in p:
    if item["id"] in ("gsm5570", "gsm4131", "gsm1398"):
        keep[item["id"]] = item
        print("=====", item["id"], "gold:", repr(item.get("gold")))
        print(item["q"][:400])
json.dump(keep, open("/home/bladless/wsp4/sandbox/analyst_a/items.json", "w"))
msgs = [{"role": "user", "content": keep["gsm5570"]["q"]}]
txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
print("PROMPT_TAIL:", repr(txt[-120:]))
ids = tok.encode(txt, add_special_tokens=False)
print("prompt_len_tokens", len(ids))
