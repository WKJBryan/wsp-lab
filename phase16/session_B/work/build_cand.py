"""Build a candidate model directory from the pristine base plus a named weight edit.
usage: python build_cand.py <name> <edit> [<edit> ...]
Creates ~/wsp14/work/cands/<name>/ as a copy of base_model with the edited weight file, and writes
<name>.json with the edit description and the tensors changed. Nothing under sandbox/ is touched.

Edits (see edits.py):  R<lam>  flip the Wait component of the copy heads;  H<layer>.<head>x<f>  scale a head.
"""
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edits as ED  # noqa: E402
from val_edit import make_edit  # noqa: E402

HOME = Path.home()
BASE = HOME / "wsp14/sandbox/base_model"
CANDS = HOME / "wsp14/work/cands"


def load_state(d):
    shards = sorted(Path(d).glob("*.safetensors"))
    assert len(shards) == 1, shards
    out = {}
    with safe_open(shards[0], framework="pt", device="cpu") as f:
        meta = f.metadata()
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return shards[0], out, (meta or {})


def main():
    name, edit_names = sys.argv[1], sys.argv[2:]
    dst = CANDS / name
    if dst.exists():
        shutil.rmtree(dst)
    CANDS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", str(BASE), str(dst)], check=True)
    subprocess.run(["chmod", "-R", "u+w", str(dst)], check=True)
    shard, state, meta = load_state(dst)
    before = {k: v.clone() for k, v in state.items() if k.startswith("model.language_model.")}
    A = ED.StateAccess(state)
    infos = [make_edit(e)(A) for e in edit_names]
    # describe the change
    changed = []
    for k in sorted(A.changed):
        full = A.pre + k
        d = (state[full].float() - before[full].float())
        changed.append(dict(tensor=full, shape=list(state[full].shape), changed_values=int((d != 0).sum()),
                            numel=d.numel(), norm_before=float(before[full].float().norm()), norm_delta=float(d.norm()),
                            max_abs_delta=float(d.abs().max())))
    save_file(state, str(shard), metadata=meta)
    desc = dict(name=name, edits=edit_names, info=infos, changed=changed)
    (CANDS / f"{name}.json").write_text(json.dumps(desc, indent=1, default=str))
    print(json.dumps(desc["changed"], indent=1))
    for k, v in state.items():
        if v.dtype != before.get(k, v).dtype if k in before else False:
            print("DTYPE CHANGED", k)


if __name__ == "__main__":
    main()
