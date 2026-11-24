"""Phase-7 pruning (protocol addendum 8): copy a base model and set whole down_proj columns to zero.

Which columns: one arm of phase7/prune_sets.json (written by prune_check.py).

Usage: uv run python phase7/build_prune.py <name> --arm spike|spike_ctrl|quiet5|quiet20|quiet50 --out <dir> [--base <model dir>]
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default=str(HERE.parent / "phase3" / "models" / "c3_original_edit"))
args = ap.parse_args()
spec = {int(L): cols for L, cols in json.loads((HERE / "prune_sets.json").read_text())[args.arm].items()}

out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(args.base, out, ignore=shutil.ignore_patterns(".cache"))
sf = sorted(out.glob("*.safetensors"))[0]
ts = load_file(str(sf))
before = {k: v.clone() for k, v in ts.items()}
key = lambda L: f"model.language_model.layers.{L}.mlp.down_proj.weight"
for L, cols in spec.items():
    ts[key(L)][:, cols] = 0
save_file(ts, str(sf), metadata={"format": "pt"})

# verify what landed on disk: the listed columns are zero and nothing else in the model changed
back = load_file(str(sf))
touched = {key(L) for L in spec}
for k, v in back.items():
    if k not in touched:
        assert (v == before[k]).all(), f"{k} changed"
for L, cols in spec.items():
    diff = (back[key(L)] != before[key(L)]).any(0).nonzero().flatten().tolist()
    assert not back[key(L)][:, cols].any() and set(diff) <= set(cols), f"L{L}: wrong columns changed"
meta = dict(name=args.name, arm=args.arm, base=Path(args.base).name, units=sum(len(c) for c in spec.values()),
            per_layer={f"L{L}": len(c) for L, c in sorted(spec.items())})
(out / "prune_meta.json").write_text(json.dumps(meta, indent=1))
(HERE / "causal" / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
print(args.name, json.dumps(meta))
