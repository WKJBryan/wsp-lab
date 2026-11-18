"""Free: load with transformers, report missing/unexpected keys."""

import torch
from transformers import AutoModelForImageTextToText, AutoConfig

MODEL = "/home/user/wsp11/sandbox/model"


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    model, info = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu", output_loading_info=True,
    )
    print("missing keys:", len(info["missing_keys"]))
    for k in sorted(info["missing_keys"])[:50]:
        print("  MISSING:", k)
    print("unexpected keys:", len(info["unexpected_keys"]))
    for k in sorted(info["unexpected_keys"])[:50]:
        print("  UNEXPECTED:", k)
    print("error msgs:", info.get("error_msgs"))
    del model


if __name__ == "__main__":
    main()
