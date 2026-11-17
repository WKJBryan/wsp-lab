"""Build the A2r matched random-edit control for attempt-2's final candidate.

Editor's edit: rows with norm < median/6 of layers {23,29} self_attn.q_proj
scaled x2 (298 + 374 rows). Control: SAME tensors, SAME counts, SAME x2 factor,
RANDOM rows (seed 20261117), from the pristine backup. Reports both delta norms.

Usage (WSL): ~/wsp3c-venv/bin/python make_a2r.py
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path.home() / "wsp11/sandbox/model_backup_orig"
DST = Path.home() / "wsp11/sandbox/model_a2r"
LAYERS = (23, 29)
COUNTS = {23: 298, 29: 374}
FACTOR = 2.0
SEED = 20261117

DST.mkdir(parents=True, exist_ok=True)
tensors = load_file(str(SRC / "model.safetensors"))
rng = torch.Generator().manual_seed(SEED)

report = {}
for li, n in COUNTS.items():
    name = f"model.language_model.layers.{li}.self_attn.q_proj.weight"
    t = tensors[name].float()
    rows = t.shape[0]
    # editor delta norm (norm<median/6 rows, x2)
    med = t.norm(dim=1).median()
    dead = t.norm(dim=1) < med / 6
    ed_delta = (t[dead] * (FACTOR - 1)).norm().item()
    ed_count = int(dead.sum())
    # control: random rows, same count, same factor
    idx = torch.randperm(rows, generator=rng)[:n]
    ctrl_delta = (t[idx] * (FACTOR - 1)).norm().item()
    t[idx] *= FACTOR
    tensors[name] = t.to(torch.bfloat16)
    report[name] = dict(editor_rows=ed_count, editor_delta_norm=round(ed_delta, 2),
                        control_rows=n, control_delta_norm=round(ctrl_delta, 2))

save_file(tensors, str(DST / "model.safetensors"), metadata={"format": "pt"})
for f in SRC.parent.glob("model_backup_orig/*"):
    if f.suffix in (".json", ".jinja") and f.name != "model.safetensors.index.json":
        shutil.copy(f, DST / f.name)
print(json.dumps(report, indent=1))
print("saved", DST)
