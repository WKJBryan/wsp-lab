"""Print module tree of Gemma4 text model on meta device (no weight loading)."""
import torch
from transformers import AutoConfig, Gemma4ForConditionalGeneration

MODEL = "/home/user/wsp11/sandbox/model"
cfg = AutoConfig.from_pretrained(MODEL)
with torch.device("meta"):
    m = Gemma4ForConditionalGeneration._from_config(cfg, torch_dtype=torch.bfloat16)

seen = set()
for name, mod in m.named_modules():
    # print unique module class paths, abbreviated
    parts = name.split(".")
    if len(parts) > 2 and parts[2] == "layers":
        # compress layer index
        if len(parts) > 3 and parts[3].isdigit():
            parts[3] = "N"
    key = ".".join(parts)
    cls = type(mod).__name__
    line = f"{key} [{cls}]"
    if line not in seen:
        seen.add(line)
        if len(parts) <= 5:
            print(line)
