"""Phase-7 causal spot-check: build a loop "brake" from OUR activation map alone.

No knowledge of the original edit's columns or direction is used. Per chosen layer L:
  d    = unit(mean resid(loop) - mean resid(pre)) after layer L        (probe_out/resid.npz)
  cols = the 64 MLP neurons with the largest loop-vs-pre effect size   (probe_out/neurons.npz)
  a    = -c * (mean_loop - mean_pre) on those neurons, c set so the push along d is
         1.5 x the loop-pre residual gap stronger inside loops than before them, times --dose
         (1.5 x gap is what the original edit does at its own layers; atlas Finding 7)
  W_down[:, cols] += outer(d, a)      -> neurons that fire more in loops subtract d
Variants: --random     same neurons and amplitudes, random direction orthogonal to d (control)
          --selective  a made orthogonal to the clean-thinking mean activation, so the
                       standing push on clean thinking is zero by construction

Usage: uv run python phase7/build_brake.py <name> --out <dir> [--layers 10,14] [--dose 1.0] [--random] [--selective]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "phase3" / "models" / "stock"
PO = HERE / "probe_out"
NCOL, GAIN = 64, 1.5

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="10,14")
ap.add_argument("--dose", type=float, default=1.0)
ap.add_argument("--random", action="store_true")
ap.add_argument("--selective", action="store_true")
# replication arms (protocol addendum 2): defaults reproduce the first batch exactly
ap.add_argument("--seed", type=int, default=1, help="seed of the random direction / random neurons")
ap.add_argument("--random-neurons", action="store_true", help="same amplitudes on 64 random non-loop neurons")
ap.add_argument("--orig-cols", action="store_true",
                help="use the ORIGINAL edit's columns and amplitudes at L15/L19 (direction u unless --random)")
args = ap.parse_args()
layers = [15, 19] if args.orig_cols else [int(x) for x in args.layers.split(",")]
ED = np.load(PO / "edit.npz")

R, NZ = np.load(PO / "resid.npz"), np.load(PO / "neurons.npz")


def neuron_stats(region):
    n = sum(int(NZ[f"cnt_{region}_{h}"]) for h in (0, 1))
    m = sum(NZ[f"sum_{region}_{h}"] for h in (0, 1)) / n
    v = np.maximum(sum(NZ[f"sq_{region}_{h}"] for h in (0, 1)) / n - m * m, 1e-12)
    return m, v  # (24, 3584)


means = {r: neuron_stats(r) for r in ("loop", "pre", "clean", "deep")}
plan, meta = {}, dict(name=args.name, layers=layers, dose=args.dose, random=args.random,
                      selective=args.selective, seed=args.seed, random_neurons=args.random_neurons,
                      orig_cols=args.orig_cols, ncol=NCOL, gain=GAIN, per_layer={})
for L in layers:
    g = R["resid_loop"][:, L + 1].astype(np.float64).mean(0) - R["resid_pre"][:, L + 1].astype(np.float64).mean(0)
    gap, d = np.linalg.norm(g), g / np.linalg.norm(g)
    (ml, vl), (mp, vp) = means["loop"], means["pre"]
    cohen = (ml[L] - mp[L]) / np.sqrt((vl[L] + vp[L]) / 2)
    cols = np.sort(np.argsort(-np.abs(cohen))[:NCOL])
    delta_act = (ml[L] - mp[L])[cols]
    a = -delta_act
    if args.selective:
        mc = means["clean"][0][L][cols]
        a = a - (a @ mc) / (mc @ mc) * mc
    a = a * (-GAIN * gap * args.dose) / (a @ delta_act)
    if args.random_neurons:  # same amplitude vector, on neurons that are NOT the loop-active ones
        pool = np.setdiff1d(np.arange(len(cohen)), cols)
        cols = np.sort(np.random.default_rng(5000 * args.seed + L).choice(pool, NCOL, replace=False))
    if args.orig_cols:
        cols, a, d = ED[f"cols{L}"], ED[f"a{L}"].astype(np.float64) * args.dose, ED[f"u{L}"].astype(np.float64)
    direction = d
    if args.random:
        r = np.random.default_rng(1000 * args.seed + L).standard_normal(1024)
        r -= (r @ d) * d
        direction = r / np.linalg.norm(r)
    drive = {reg: round(float(a @ means[reg][0][L][cols]), 3) for reg in means}
    if not (args.random_neurons or args.orig_cols):
        assert abs((drive["loop"] - drive["pre"]) + GAIN * gap * args.dose) < 0.02 * GAIN * gap * args.dose + 0.01
    plan[L] = (cols, np.outer(direction, a))
    meta["per_layer"][f"L{L}"] = dict(resid_gap=round(float(gap), 3), cols=cols.tolist(),
                                      a=[round(float(x), 5) for x in a], drive=drive,
                                      fro=round(float(np.linalg.norm(plan[L][1])), 4),
                                      cos_dir_d=round(float(direction @ d), 4))

out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(STOCK, out, ignore=shutil.ignore_patterns(".cache"))
sf = sorted(out.glob("*.safetensors"))[0]
ts = load_file(str(sf))
stock = {}
for L, (cols, delta) in plan.items():
    k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    stock[k] = ts[k].clone()
    w = ts[k].float()
    w[:, cols] += torch.from_numpy(delta).float()
    ts[k] = w.to(stock[k].dtype)
save_file(ts, str(sf), metadata={"format": "pt"})

# verify what actually landed on disk (bf16 rounding included)
back = load_file(str(sf))
for L, (cols, delta) in plan.items():
    k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    real = back[k].float() - stock[k].float()
    untouched = np.setdiff1d(np.arange(real.shape[1]), cols)
    assert not real[:, untouched].any(), f"L{L}: columns outside the plan changed"
    err = float((real[:, cols] - torch.from_numpy(delta).float()).norm() / np.linalg.norm(delta))
    assert err < 0.05, f"L{L}: realised delta is off by {err:.3f} (bf16 rounding too coarse?)"
    meta["per_layer"][f"L{L}"].update(realised_rel_err=round(err, 4), changed_values=int((real != 0).sum()))
(out / "brake_meta.json").write_text(json.dumps(meta, indent=1))
(HERE / "causal").mkdir(exist_ok=True)
(HERE / "causal" / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
print(args.name, json.dumps({k: {x: v[x] for x in ("resid_gap", "drive", "fro", "cos_dir_d", "realised_rel_err", "changed_values")}
                             for k, v in meta["per_layer"].items()}))
