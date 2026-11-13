"""Phase-7 atlas — interactive "light map": which MLP neurons respond to which job, per layer.

Packs one effect size (Cohen's d) per neuron per job into phase7/lightmap.html, a
self-contained page built from phase7/lightmap_template.html.

Jobs and where their numbers come from (all stock model, probe_out/*.npz):
  Stuck in a loop   in-loop vs before-the-loop positions, while GENERATING   (neurons.npz)
  7 MATH subjects   that subject vs the other six, while READING the question (questions.npz)
  Hard question     level 4-5 vs level 1-2, while READING the question        (questions.npz)

Usage: uv run python phase7/build_lightmap.py
"""

import base64
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
PO = HERE / "probe_out"
SCALE = 40  # int8 = round(d * 40), so |d| is stored up to 3.17

NZ = np.load(PO / "neurons.npz")


def gen_stats(region):
    n = sum(int(NZ[f"cnt_{region}_{h}"]) for h in (0, 1))
    m = sum(NZ[f"sum_{region}_{h}"] for h in (0, 1)) / n
    return m, np.maximum(sum(NZ[f"sq_{region}_{h}"] for h in (0, 1)) / n - m * m, 1e-12)


def cohen(a, b):  # a, b: (items, 24, 3584)
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)


(ml, vl), (mp, vp) = gen_stats("loop"), gen_stats("pre")
jobs = [("Stuck in a loop", (ml - mp) / np.sqrt((vl + vp) / 2))]

Q = np.load(PO / "questions.npz")["neurons"].astype(np.float32)
ds = load_from_disk(str(HERE.parent / "phase3" / "data" / "math500"))
subject, level = np.array(ds["subject"]), np.array(ds["level"])
for s in sorted(set(subject)):
    jobs.append((s, cohen(Q[subject == s], Q[subject != s])))
jobs.append(("Hard question", cohen(Q[level >= 4], Q[level <= 2])))

# trunc, not rint: a unit at 0.79 must not count as "on" at the 0.8 slider step
packed = np.stack([np.clip(np.trunc(d.astype(np.float64) * SCALE), -127, 127).astype(np.int8) for _, d in jobs])  # (9, 24, 3584)
for q, (name, d) in zip(packed, jobs):  # self-check: page counts (|int8| >= thr*40) equal the true counts at every slider step
    for t in range(5, 21):
        page, true = int((np.abs(q.astype(int)) >= t * SCALE // 10).sum()), int((np.abs(d.astype(np.float64)) >= t / 10).sum())
        assert page == true, (name, t / 10, page, true)
ED = np.load(PO / "edit.npz")
brake = json.loads((HERE / "causal" / "brake_d100.json").read_text())["per_layer"]
data = dict(
    jobs=[n for n, _ in jobs], layers=24, neurons=3584, scale=SCALE,
    b64=base64.b64encode(packed.tobytes()).decode(),
    marks=dict(original={"15": ED["cols15"].tolist(), "19": ED["cols19"].tolist()},
               brake={k[1:]: v["cols"] for k, v in brake.items()}),
)
html = (HERE / "lightmap_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/", "const DATA = " + json.dumps(data) + ";")
(HERE / "lightmap.html").write_text(html, encoding="utf-8")
for k, (name, d) in enumerate(jobs):
    on = (np.abs(d) >= 0.8).sum(1)
    print(f"{name:24s} lights on (|d|>=0.8): {int(on.sum()):6d} | busiest floors {np.argsort(-on)[:3].tolist()}")
print(f"lightmap.html: {len(html) / 1e6:.2f} MB")
