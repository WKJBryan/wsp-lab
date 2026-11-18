"""A2r control for attempt 5: MiMo-pro's edit was uniform x0.4 on the FINAL
RMSNorm gain (model.language_model.norm.weight). Control: same operation class
and factor on a randomly chosen per-layer norm tensor (seed 20261118), from the
pristine wound. Usage (WSL): ~/wsp3c-venv/bin/python make_a2r5.py
"""

import json
import random
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path.home() / "wsp11/gemma4-8exp"          # pristine wound
DST = Path.home() / "wsp11/sandbox/model_a2r5"
FACTOR = 0.4
rng = random.Random(20261118)

tensors = load_file(str(SRC / "model.safetensors"))
norm_tensors = sorted(k for k in tensors
                      if ".layernorm" in k or k.endswith("model.language_model.norm.weight"))
pick = norm_tensors[rng.randrange(len(norm_tensors))]
before = tensors[pick].float().norm().item()
tensors[pick] = (tensors[pick].float() * FACTOR).to(torch.bfloat16)
after = tensors[pick].float().norm().item()

DST.mkdir(parents=True, exist_ok=True)
save_file(tensors, str(DST / "model.safetensors"), metadata={"format": "pt"})
for f in SRC.glob("*"):
    if f.suffix in (".json", ".jinja") and f.name != "model.safetensors.index.json":
        shutil.copy(f, DST / f.name)
print(json.dumps(dict(picked=pick, norm_before=round(before, 2),
                      norm_after=round(after, 2), factor=FACTOR), indent=1))
