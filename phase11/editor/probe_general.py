"""Scope check: non-math generation quality (story, factual QA, code)."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

PROMPTS = [
    "Write a short story about a fox who learns to swim.",
    "What is the capital of France? Answer in one word.",
    "Write a Python function that adds two numbers.",
    "Explain photosynthesis in two sentences.",
    "List three colors.",
]

for q in PROMPTS:
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to("cuda")
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=150, pad_token_id=0)
    t = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"\n=== {q}\n{t[:600]}")
