"""Phase-7 atlas — interactive "light map": which MLP neurons respond to which job, per layer.

Packs one effect size (Cohen's d) per neuron per job into phase7/lightmap.html, a
self-contained page built from phase7/lightmap_template.html.

Jobs: see phase7/jobs.py. Labels on hover: phase7/labels.csv (label_units.py) and phase7/knockout.json. Wires between floors: phase7/wires.json, from build_wires.py (run it first).

Usage: uv run python phase7/build_lightmap.py
"""

import base64
import csv
import json
from pathlib import Path

import numpy as np

from jobs import job_stats

HERE = Path(__file__).resolve().parent
PO = HERE / "probe_out"
SCALE = 40  # int8 = round(d * 40), so |d| is stored up to 3.17

jobs = [(name, shift / sd) for name, shift, sd, _ in job_stats()]
W = json.loads((HERE / "wires.json").read_text())
assert W["jobs"] == [n for n, _ in jobs], "wires.json was built for a different job list"
RM = json.loads((HERE / "reading_remeasure.json").read_text())["jobs"]
hubs = {}  # hub test results (test_hubs.py), kept only where the test still describes the shipped wires
for k, h in json.loads((HERE / "hubs.json").read_text()).items():
    job = k.split(" | ")[1]
    if sum(w[:2] == h["hub"] for w in W["wires"][job]) == h["wires"]["targets"]:
        hubs[job] = dict(unit=f"floor {h['hub'][0]} unit {h['hub'][1]}", same=h["wires"]["same_direction"],
                         targets=h["wires"]["targets"], ratio=h["wires"]["median_measured_over_predicted"])

# trunc, not rint: a unit at 0.79 must not count as "on" at the 0.8 slider step
packed = np.stack([np.clip(np.trunc(d.astype(np.float64) * SCALE), -127, 127).astype(np.int8) for _, d in jobs])  # (9, 24, 3584)
for q, (name, d) in zip(packed, jobs):  # self-check: page counts (|int8| >= thr*40) equal the true counts at every slider step
    for t in range(5, 21):
        page, true = int((np.abs(q.astype(int)) >= t * SCALE // 10).sum()), int((np.abs(d.astype(np.float64)) >= t / 10).sum())
        assert page == true, (name, t / 10, page, true)
# labels shown on hover: one bit per tag, from label_units.py; "held" = subjects whose label passed the knockout test
TAGS = ["first-word spike", "loop marker", "question length"] + [f"subject: {n}" for n, _ in jobs[1:8]]
lab = np.zeros((24, 3584), np.uint16)
with open(HERE / "labels.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        for t in r["all_tags"].split("; "):
            if t in TAGS:
                lab[int(r["floor"]), int(r["unit"])] |= 1 << TAGS.index(t)
LS = json.loads((HERE / "labels_summary.json").read_text())
for i, t in enumerate(TAGS):  # self-check: the packed bits count the same units as label_units.py did
    want = LS["each_subject"][t[9:]] if t.startswith("subject: ") else LS["units_with_tag"][t]
    assert int((lab >> i & 1).sum()) == want, (t, int((lab >> i & 1).sum()), want)
for i, (n, d) in enumerate(jobs[1:8]):  # a subject label is exactly a light at 0.8 for that subject
    assert ((np.abs(d) >= 0.8) == (lab >> 3 + i & 1).astype(bool)).all(), n
KO = json.loads((HERE / "knockout.json").read_text())
PB = KO["posthoc_bootstrap"]
held = [[n, s] for n, s, a, c in zip(KO["subjects"], KO["selectivity"]["knockout"], PB["chance_selectivity_above_1"], PB["chance_above_control"])
        if a >= 0.95 and c >= 0.95]  # reliably above 1 and above the random-lights control
ED = np.load(PO / "edit.npz")
brake = json.loads((HERE / "causal" / "brake_d100.json").read_text())["per_layer"]
data = dict(
    jobs=[n for n, _ in jobs], layers=24, neurons=3584, scale=SCALE,
    b64=base64.b64encode(packed.tobytes()).decode(),
    marks=dict(original={"15": ED["cols15"].tolist(), "19": ED["cols19"].tolist()},
               brake={k[1:]: v["cols"] for k, v in brake.items()}),
    wires=[W["wires"][n] for n, _ in jobs],  # per job: [floor, unit, higher floor, unit, contribution in d units]
    wirecheck=dict(nudge=W["check_nudge"], sum=[W["check_sum"][n]["lit_units"] for n, _ in jobs],
                   hubs=hubs),
    # per job: lights at 0.8 before prompt length was regressed out, and how many of those remain (remeasure_reading.py); None for the loop job
    labels=dict(tags=TAGS, b64=base64.b64encode(lab.astype('<u2').tobytes()).decode(), held=held, tagged=LS["tagged"], units=LS["units"], ko_units=KO["units_per_set"]),
    lengthcheck=[RM.get(n) and dict(lit=RM[n]["lit_after"], stay=RM[n]["after_stay_length_regressed"]) for n, _ in jobs],
)
html = (HERE / "lightmap_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/", "const DATA = " + json.dumps(data) + ";")
(HERE / "lightmap.html").write_text(html, encoding="utf-8")
for k, (name, d) in enumerate(jobs):
    on = (np.abs(d) >= 0.8).sum(1)
    print(f"{name:24s} lights on (|d|>=0.8): {int(on.sum()):6d} | busiest floors {np.argsort(-on)[:3].tolist()}")
print("labels passing the knockout test:", held)
print(f"lightmap.html: {len(html) / 1e6:.2f} MB")
