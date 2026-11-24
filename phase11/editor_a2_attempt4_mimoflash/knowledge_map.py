"""Free: map preserved vs destroyed knowledge via chat-format answer logprobs.
Also: hidden-state probing — is 'Paris' direction present in the residual stream
before the final norm (readout damage) or absent (knowledge damage)?
"""

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"

# (question, correct answer, wrong answer)
QA = [
    ("Name one primary color.", " Red", " Purple"),
    ("What is 2 + 2?", " 4", " 5"),
    ("What is 3 + 4?", " 7", " 8"),
    ("What is 10 - 3?", " 7", " 6"),
    ("What is 5 * 5?", " 25", " 30"),
    ("What is the capital of France?", " Paris", " London"),
    ("What is the capital of Japan?", " Tokyo", " Kyoto"),
    ("How many days are in a week?", " 7", " 5"),
    ("How many continents are there?", " 7", " 5"),
    ("What is the largest planet?", " Jupiter", " Saturn"),
    ("What gas do humans need to breathe?", " oxygen", " helium"),
    ("What is H2O commonly known as?", " water", " acid"),
    ("Who wrote Romeo and Juliet?", " Shakespeare", " Dickens"),
    ("What is the square root of 16?", " 4", " 5"),
    ("What is 100 / 4?", " 25", " 20"),
    ("What is 12 * 12?", " 144", " 124"),
    ("Name a season of the year.", " summer", " winter"),
    ("What is the opposite of hot?", " cold", " warm"),
    ("How many legs does a spider have?", " 8", " 6"),
    ("What color is the sky on a clear day?", " blue", " green"),
]


def chat_logprob(model, tok, q, ans):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    ids = {k: v.to(model.device) for k, v in ids.items()}
    pref = ids["input_ids"].shape[1]
    c = tok.encode(ans, add_special_tokens=False)
    full = torch.cat([ids["input_ids"], torch.tensor([c]).to(model.device)], dim=1)
    with torch.no_grad():
        logits = model(full).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(pref - 1, len(full[0]) - 1):
        total += lp[i, full[0, i + 1]].item()
    return total


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    print("=== answer logprob: correct vs plausible-wrong ===")
    wins = 0
    for q, good, bad in QA:
        lg = chat_logprob(model, tok, q, good)
        lb = chat_logprob(model, tok, q, bad)
        mark = "GOOD" if lg > lb else "bad "
        if lg > lb:
            wins += 1
        print(f"  [{mark}] {q[:48]:50s} good={lg:7.2f} bad={lb:7.2f} "
              f"margin={lg - lb:+7.2f}")
    print(f"  -> model prefers correct answer in {wins}/{len(QA)}")

    # hidden-state probing for Paris
    print("\n=== hidden state direction probe ('The capital of France is') ===")
    prompt = "The capital of France is"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hs = out.hidden_states  # embed + per layer (pre-final-norm for last?)
    emb = model.model.language_model.embed_tokens.weight.float()
    targets = {}
    for word in [" Paris", " London", " Berlin", " France", " city"]:
        tid = tok.encode(word, add_special_tokens=False)[0]
        targets[word] = emb[tid] / emb[tid].norm()
    print("  layer | " + " | ".join(f"{w.strip():>7}" for w in targets))
    for i in [0, 5, 10, 15, 20, 25, 29, 30]:
        if i >= len(hs):
            continue
        h = hs[i][0, -1].float()
        hn = h / (h.norm() + 1e-9)
        sims = {w: (hn @ t).item() for w, t in targets.items()}
        print(f"  h{i:02d}  | " + " | ".join(f"{sims[w]:+7.4f}" for w in targets))


if __name__ == "__main__":
    main()
