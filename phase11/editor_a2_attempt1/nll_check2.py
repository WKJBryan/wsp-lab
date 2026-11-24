"""NLL with proper chat prefix (assistant channel) for the same texts."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

PREFIX = "<bos><|turn>user\nContinue this text exactly.\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

TEXTS = {
    "wiki_style": ("The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon biome "
                   "that covers most of the Amazon basin of South America. This basin encompasses "
                   "7,000,000 square kilometers, of which 5,500,000 square kilometers are covered by "
                   "the rainforest."),
    "story": ("Maria opened the old wooden door and stepped into the attic. Dust floated through a shaft "
              "of afternoon light. In the corner sat a trunk she had never noticed before."),
    "math_solution": ("To add the fractions $\\frac{1}{3}$ and $\\frac{1}{4}$, we find a common denominator. "
                      "The least common multiple of 3 and 4 is 12. Adding gives $\\frac{7}{12}$. "
                      "The answer is $\\boxed{\\frac{7}{12}}$."),
}

pfx_ids = tok(PREFIX, add_special_tokens=False)["input_ids"]
print("prefix len:", len(pfx_ids))

for name, text in TEXTS.items():
    body = tok(text, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([pfx_ids + body]).to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    n = len(tgt) - 0
    nll_body = -logp[len(pfx_ids) - 1: len(pfx_ids) - 1 + len(body), torch.tensor(body)].mean().item()
    top1_body = (logits[len(pfx_ids) - 1: len(pfx_ids) - 1 + len(body)].argmax(-1) == torch.tensor(body).to("cuda")).float().mean().item()
    print(f"{name:<14} body_tokens={len(body):<4} NLL={nll_body:.3f} PPL={2.71828**nll_body:.1f} top1_acc={top1_body:.3f}")
