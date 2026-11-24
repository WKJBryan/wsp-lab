"""In-memory diagnostics with transformers (free, uncounted).

Loads the sandbox model once, runs a forward pass on a prompt, reports:
  - pre-softcap logit stats (max abs, top1/top2, entropy after softcap)
  - per-layer residual-stream RMS trajectory
  - greedy generation for parity check with vLLM

Usage: ~/wsp3c-venv/bin/python diag.py
"""

import json
import math

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True,
    )
    model.eval()

    prompt = "What is 17 * 23? Show your work briefly and give the final number."
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    logits = out.logits[0, -1].float()  # pre-softcap (softcap applied in compute_logits? check)
    print("=== raw last-position logits ===")
    print("max", logits.max().item(), "min", logits.min().item(),
          "argmax", logits.argmax().item(), repr(tok.decode([logits.argmax().item()])))

    # softcap 30: applied inside model? try both interpretations
    capped = 30.0 * torch.tanh(logits / 30.0)
    probs = torch.softmax(capped, dim=-1)
    topv, topi = torch.topk(probs, 10)
    print("=== after tanh softcap 30 ===")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  {v:.6f}  {i}  {tok.decode([i])!r}")

    hs = out.hidden_states  # tuple: embed + one per layer
    print("=== residual stream RMS per layer ===")
    for i, h in enumerate(hs):
        r = h[0, -1].float().pow(2).mean().sqrt().item()
        print(f"  h{i:02d}: {r:.6g}")
    # final norm applied inside model? hidden_states last entry is post-final-norm?
    # In HF, hidden_states[-1] is after final norm for most models.

    # per-layer output delta norms: ||h_{i+1} - h_i||
    print("=== per-layer delta RMS (last pos) ===")
    for i in range(len(hs) - 1):
        d = (hs[i + 1][0, -1].float() - hs[i][0, -1].float())
        print(f"  L{i:02d} delta_rms={d.pow(2).mean().sqrt().item():.6g}")

    # greedy generation parity
    print("=== greedy generation (200 tokens) ===")
    g = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                       pad_token_id=tok.pad_token_id or 0)
    print(tok.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
