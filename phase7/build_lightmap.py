"""Phase-7 atlas — interactive "light map": which MLP neurons respond to which job, per layer.

Packs one effect size (Cohen's d) per neuron per job into phase7/lightmap.html, a
self-contained page built from phase7/lightmap_template.html.

Jobs: see phase7/jobs.py. A tenth layer, "Lever", is not a job: see lever() below. Labels on hover: phase7/labels.csv (label_units.py) and phase7/knockout.json. Wires between floors: phase7/wires.json, from build_wires.py (run it first).

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

STATS = job_stats()
jobs = [(name, shift / sd) for name, shift, sd, _ in STATS]
ED = np.load(PO / "edit.npz")
brake = json.loads((HERE / "causal" / "brake_d100.json").read_text())["per_layer"]
LEVER_PCT = 5  # one slider unit of the lever layer = 5% of the stream's clean-thinking spread along the direction


def lever():
    """(push (24, 3584), facts for the page). Not a job: how hard each unit pushes the residual stream along the
    published edit's direction u when the model is stuck, compared with before the loop. Direct write only.

    push = (loop job's mean shift of the unit) x (its down_proj column . u) / (clean-thinking SD of the stream
    along u after that floor); u = u15 for floors 0-15, u19 above. Positive = toward the stuck side of u (the
    edit itself pushes the other way). Descriptive: no prediction was made before measuring.
    """
    import torch
    from safetensors import safe_open

    shift = STATS[0][1]  # "Stuck in a loop": in-loop minus before-the-loop activity
    A3 = json.loads((HERE / "atlas_activation.json").read_text())["A3_edit_selectivity"]
    for L in (15, 19):  # self-check: activations and edit columns line up — the edit's own drive (Finding 7) is reproduced
        got, want = float(ED[f"a{L}"] @ shift[L][ED[f"cols{L}"]]), A3[f"L{L}"]["mean_drive"]["loop"] - A3[f"L{L}"]["mean_drive"]["pre"]
        assert abs(got - want) < 0.01, (L, got, want)
    clean = np.load(PO / "resid.npz")["resid_clean"]  # (positions, 25 boundaries, 1024)
    z = np.empty((24, 3584))
    with safe_open(next((HERE.parent / "phase3" / "models" / "stock").glob("*.safetensors")), "pt") as f:
        for L in range(24):
            u = ED["u15" if L <= 15 else "u19"].astype(np.float64)
            assert abs(np.linalg.norm(u) - 1) < 1e-3
            down = f.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").to(torch.float64).numpy()  # (1024, 3584)
            z[L] = shift[L] * (u @ down) / (clean[:, L + 1].astype(np.float64) @ u).std()
    top64 = lambda v, cols: len(set(np.argsort(-np.abs(v))[:64].tolist()) & set(np.asarray(cols).tolist()))
    r2 = lambda x: round(float(x), 2)
    tf, tu = (int(i) for i in np.unravel_index(np.abs(z).argmax(), z.shape))
    facts = dict(
        top=[tf, tu, round(float(abs(z[tf, tu])) * 100)],  # strongest single pusher: floor, unit, % of the clean spread (the page's int8 clips at 15.9%)
        job=len(jobs), pct=LEVER_PCT, net=[r2(x) for x in z.sum(1)],  # net push per floor, in SDs of the clean spread
        gross=[r2(x) for x in np.abs(z).sum(1)],  # the same with every unit's push counted as positive: how much cancels
        # the published edit's 64 units per floor: how many the lever ranking finds vs the loop-light ranking; how many push toward stuck; their summed push
        edit={str(L): dict(by_lever=top64(z[L], ED[f"cols{L}"]), by_light=top64(jobs[0][1][L], ED[f"cols{L}"]),
                           toward=int((z[L][ED[f"cols{L}"]] > 0).sum()), push=r2(z[L][ED[f"cols{L}"]].sum())) for L in (15, 19)},
        brake={k[1:]: r2(z[int(k[1:])][v["cols"]].sum()) for k, v in brake.items()})  # summed push of the map's test-brake units
    return z * 100 / LEVER_PCT, facts


W = json.loads((HERE / "wires.json").read_text())
assert W["jobs"] == [n for n, _ in jobs], "wires.json was built for a different job list"
RM = json.loads((HERE / "reading_remeasure.json").read_text())["jobs"]
hubs = {}  # hub test results (test_hubs.py), kept only where the test still describes the shipped wires
for k, h in json.loads((HERE / "hubs.json").read_text()).items():
    job = k.split(" | ")[1]
    if sum(w[:2] == h["hub"] for w in W["wires"][job]) == h["wires"]["targets"]:
        hubs[job] = dict(unit=f"floor {h['hub'][0]} unit {h['hub'][1]}", same=h["wires"]["same_direction"],
                         targets=h["wires"]["targets"], ratio=h["wires"]["median_measured_over_predicted"])

lever_d, lever_facts = lever()
layers = jobs + [("Lever (deep fix direction)", lever_d)]  # what the page can show: the nine jobs, then the lever layer
# trunc, not rint: a unit at 0.79 must not count as "on" at the 0.8 slider step
packed = np.stack([np.clip(np.trunc(d.astype(np.float64) * SCALE), -127, 127).astype(np.int8) for _, d in layers])  # (10, 24, 3584)
for q, (name, d) in zip(packed, layers):  # self-check: page counts (|int8| >= thr*40) equal the true counts at every slider step
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
data = dict(
    jobs=[n for n, _ in layers], layers=24, neurons=3584, scale=SCALE, lever=lever_facts,
    b64=base64.b64encode(packed.tobytes()).decode(),
    marks=dict(original={"15": ED["cols15"].tolist(), "19": ED["cols19"].tolist()},
               brake={k[1:]: v["cols"] for k, v in brake.items()}),
    wires=[W["wires"][n] for n, _ in jobs] + [[]],  # per job: [floor, unit, higher floor, unit, contribution in d units]; none for the lever layer
    wirecheck=dict(nudge=W["check_nudge"], sum=[W["check_sum"][n]["lit_units"] for n, _ in jobs] + [None],
                   hubs=hubs),
    # per job: lights at 0.8 before prompt length was regressed out, and how many of those remain (remeasure_reading.py); None for the loop job
    labels=dict(tags=TAGS, b64=base64.b64encode(lab.astype('<u2').tobytes()).decode(), held=held, tagged=LS["tagged"], units=LS["units"], ko_units=KO["units_per_set"]),
    lengthcheck=[RM.get(n) and dict(lit=RM[n]["lit_after"], stay=RM[n]["after_stay_length_regressed"]) for n, _ in layers],
)
html = (HERE / "lightmap_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/", "const DATA = " + json.dumps(data) + ";")
(HERE / "lightmap.html").write_text(html, encoding="utf-8")
for k, (name, d) in enumerate(layers):
    on = (np.abs(d) >= 0.8).sum(1)
    print(f"{name:24s} lights on (|d|>=0.8): {int(on.sum()):6d} | busiest floors {np.argsort(-on)[:3].tolist()}")
print("lever layer:", lever_facts)
print("labels passing the knockout test:", held)
print(f"lightmap.html: {len(html) / 1e6:.2f} MB")
