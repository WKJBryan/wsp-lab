"""Probability profile: teacher-force clean answers; measure p(top1), p(true token), entropy;
also free-running p(top1) trajectory. Detects global logit flattening."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

QA = [
    ("Compute $17\\times 23 - 11$.",
     "To compute $17 \\times 23 - 11$:\n\n**Step 1: Multiplication.** $17 \\times 23 = 391$.\n\n"
     "**Step 2: Subtraction.** $391 - 11 = 380$.\n\nThe answer is $\\boxed{380}$."),
    ("What is $1+2+3+\\cdots+100$?",
     "The sum $1+2+3+\\cdots+100$ is an arithmetic series. Using the formula, the sum is "
     "$\\frac{100 \\cdot 101}{2} = 5050$. The answer is $\\boxed{5050}$."),
    ("What is the greatest common divisor of 84 and 144?",
     "Factor: $84 = 2^2 \\cdot 3 \\cdot 7$ and $144 = 2^4 \\cdot 3^2$. The common factors give "
     "$\\gcd = 2^2 \\cdot 3 = 12$. The answer is $\\boxed{12}$."),
]

print("=== teacher-forced probability profile on clean solutions ===")
for q, a in QA:
    enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  tokenize=True, return_tensors="pt", return_dict=True)
    pfx = enc["input_ids"]
    ans_ids = tok(a, add_special_tokens=False)["input_ids"]
    full = torch.cat([pfx, torch.tensor([ans_ids])], dim=1).to("cuda")
    with torch.no_grad():
        logits = model(full).logits[0].float()
    probs = torch.softmax(logits, -1)
    p1 = probs.max(-1).values
    ptops, ptrues, ents = [], [], []
    for t in range(pfx.shape[1] - 1, full.shape[1] - 1):
        true_tok = int(full[0, t + 1])
        ptops.append(p1[t].item())
        ptrues.append(probs[t, true_tok].item())
        ents.append(float(-(probs[t] * (probs[t] + 1e-12).log()).sum()))
    import statistics as st
    print(f"\nQ: {q[:50]}  (n={len(ptops)} answer tokens)")
    print(f"  p(top1): mean={st.mean(ptops):.3f} min={min(ptops):.3f}  frac>0.5={sum(x>0.5 for x in ptops)/len(ptops):.2f}")
    print(f"  p(true): mean={st.mean(ptrues):.3f} min={min(ptrues):.4f}")
    print(f"  entropy: mean={st.mean(ents):.2f} max={max(ents):.2f} (uniform over 262144 = 12.5)")

print("\n=== logit scale check (pre-softcap top logits at last positions) ===")
with torch.no_grad():
    logits = model(full).logits[0, -1].float()
print("top-5 raw logits:", [round(v, 1) for v in torch.topk(logits, 5).values.tolist()])
