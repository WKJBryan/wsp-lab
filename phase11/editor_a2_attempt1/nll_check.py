"""Teacher-forced NLL on general text + math text: absolute LM quality check."""
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
tok = AutoTokenizer.from_pretrained(MODEL)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

TEXTS = {
    "wiki_style": ("The Amazon rainforest, also called Amazon jungle or Amazonia, is a moist broadleaf "
                   "tropical rainforest in the Amazon biome that covers most of the Amazon basin of South "
                   "America. This basin encompasses 7,000,000 square kilometers, of which 5,500,000 square "
                   "kilometers are covered by the rainforest. This region includes territory belonging to "
                   "nine nations and 3,344 formally acknowledged indigenous territories."),
    "story": ("Maria opened the old wooden door and stepped into the attic. Dust floated through a shaft "
              "of afternoon light. In the corner sat a trunk she had never noticed before, its brass "
              "lock green with age. She knelt beside it, lifted the lid, and found letters tied with a "
              "blue ribbon, each one addressed to her grandmother."),
    "math_solution": ("To add the fractions $\\frac{1}{3}$ and $\\frac{1}{4}$, we find a common denominator. "
                      "The least common multiple of 3 and 4 is 12, so $\\frac{1}{3} = \\frac{4}{12}$ and "
                      "$\\frac{1}{4} = \\frac{3}{12}$. Adding gives $\\frac{4}{12} + \\frac{3}{12} = "
                      "\\frac{7}{12}$. The answer is $\\boxed{\\frac{7}{12}}$."),
    "loop_text": ("The expression is written as:\n$$1+2+3+\\dots+100$$\n\nThe expression is written as:\n"
                  "$$1+2+3+\\dots+100$$\n\nThe expression is written as:\n$$1+2+3+\\dots+100$$\n\n"),
}

for name, text in TEXTS.items():
    ids = tok("<bos>" + text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    nll = -logp[torch.arange(len(tgt)), tgt].mean().item()
    acc1 = (logits.argmax(-1)[1:] == tgt).float().mean().item()
    print(f"{name:<14} tokens={len(tgt):<4} NLL={nll:.3f} (PPL={2.818**nll * 0 + 2**nll if False else 2.71828**nll:.1f}) top1_acc={acc1:.3f}")
