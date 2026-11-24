"""Phase-7 atlas — "wires": which MLP unit on a lower floor directly feeds which unit on a higher floor.

DIRECT path only: unit i writes its down_proj column into the residual stream; unit j on a later
floor reads the stream through its RMSNorm and gate_proj / up_proj. Attention and the MLPs in
between are not followed, so a wire is "the straight cable", not "everything i does to j".

  wire[j, i]   = d act_j / d act_i, averaged over positions
               = mean( silu'(g_j) u_j / rms ) * (gate_j * (1 + w_norm)) . down_i
               + mean( silu(g_j)      / rms ) * (up_j   * (1 + w_norm)) . down_i
                 operating point: stock model READING the 500 MATH-500 prompts, every position
  contribution = wire[j, i] * (job's mean shift of i) / (job's SD of j)
                 -> light-map units: how much of j's Cohen's d this one wire accounts for

Checks, all written to wires.json:
  formulas  the SwiGLU / RMSNorm formulas above reproduce the model's own activations (assert)
  sum       per job: do the summed wire contributions line up with the measured d of each unit?
  full      same sum, but over EVERYTHING the unit reads (words, attention and lower MLPs, taken from the
            measured shift of the residual stream): tells a weak "sum" apart from a broken linear model
  nudge     raise unit i by 1 SD in a real forward pass; compare unit j's measured change with wire * SD

shortcut: one operating point (reading) for every job, including the loop job, which was
measured while generating; re-measure inside loops if the loop wires are to carry a claim.

Outputs: phase7/wires.json
Usage: uv run python phase7/build_wires.py      (then build_lightmap.py, which embeds it)
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jobs import job_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOP, LIT = 600, 0.5        # wires kept per job; both ends must be lit at the page's lowest slider step
NUDGE_TOP, NUDGE_PROMPTS = 15, 8
dev = "cuda"

stock = ROOT / "phase3" / "models" / "stock"
tok = AutoTokenizer.from_pretrained(stock)
model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.float32).eval().to(dev)
mods = dict(model.named_modules())
pre = next(n for n in mods if n.endswith("layers.0.mlp"))[: -len("0.mlp")]
NL = 24
norm = [mods[f"{pre}{L}.post_attention_layernorm"] for L in range(NL)]
gate = [mods[f"{pre}{L}.mlp.gate_proj"].weight for L in range(NL)]   # (3584, 1024)
up = [mods[f"{pre}{L}.mlp.up_proj"].weight for L in range(NL)]
down = [mods[f"{pre}{L}.mlp.down_proj"] for L in range(NL)]          # weight (1024, 3584)
NU = gate[0].shape[0]

problems = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))


def ids_of(p):
    prompt = tok.apply_chat_template([{"role": "user", "content": p["q"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    return tok(prompt, return_tensors="pt").input_ids.to(dev)


# ---- pass 1: operating point --------------------------------------------------------------
acc = {k: torch.zeros(NL, NU, dtype=torch.float64, device=dev) for k in ("A", "B", "s1", "s2")}
state = dict(n=0, checked=0, mode="stats", nudge=None, seen={})
normed = {}


def norm_hook(L):
    def fn(m, inp, out):
        x = inp[0][0]
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        if state["checked"] < NL:  # formula check: this model's RMSNorm is (1 + w) * x / rms(x)
            err = float((out[0] - (1 + m.weight) * x / rms).norm() / out[0].norm())
            assert err < 1e-3, f"L{L}: RMSNorm is not (1+w)*x/rms (rel err {err:.4f})"
        normed[L] = (out[0], rms)
    return fn


def down_hook(L):
    def fn(m, inp):
        a = inp[0]
        if state["mode"] == "stats":
            y, rms = normed[L]
            g, u = y @ gate[L].T, y @ up[L].T
            sg = torch.sigmoid(g)
            if state["checked"] < NL:  # formula check: the unit's activation is silu(gate) * up
                err = float((F.silu(g) * u - a[0]).norm() / a[0].norm())
                assert err < 1e-3, f"L{L}: MLP is not silu(gate)*up (rel err {err:.4f})"
                state["checked"] += 1
            acc["A"][L] += (sg * (1 + g * (1 - sg)) * u / rms).sum(0).double()
            acc["B"][L] += (F.silu(g) / rms).sum(0).double()
            acc["s1"][L] += a[0].sum(0).double()
            acc["s2"][L] += a[0].pow(2).sum(0).double()
            return None
        state["seen"][L] = a[0].mean(0)  # nudge mode: mean activation of every unit on this floor
        if state["nudge"] and state["nudge"][0] == L:
            a = a.clone()
            a[..., state["nudge"][1]] += state["nudge"][2]
            return (a,)
    return fn


for L in range(NL):
    norm[L].register_forward_hook(norm_hook(L))
    down[L].register_forward_pre_hook(down_hook(L))

with torch.no_grad():
    for p in problems:
        ids = ids_of(p)
        model(ids)
        state["n"] += ids.shape[1]
n = state["n"]
A, B = (acc["A"] / n).float(), (acc["B"] / n).float()
sd_pos = (acc["s2"] / n - (acc["s1"] / n) ** 2).clamp_min(1e-12).sqrt().float()  # per-position SD of each unit
read = [(A[L][:, None] * gate[L] + B[L][:, None] * up[L]) * (1 + norm[L].weight)[None, :] for L in range(NL)]
print(f"operating point from {len(problems)} prompts, {n} positions; formula checks passed on {state['checked']} floors")

# ---- wires and per-job contributions ------------------------------------------------------
jobs = job_stats()
names = [j[0] for j in jobs]
shift = torch.tensor(np.stack([j[1] for j in jobs]), dtype=torch.float32, device=dev)  # (9, 24, 3584)
sd = torch.tensor(np.stack([j[2] for j in jobs]), dtype=torch.float32, device=dev)
d = shift / sd
lit = d.abs() >= LIT
rshift = torch.tensor(np.stack([j[3] for j in jobs]), dtype=torch.float32, device=dev)  # (9, 25, 1024)
ED = np.load(HERE / "probe_out" / "edit.npz")
brake = json.loads((HERE / "causal" / "brake_d100.json").read_text())["per_layer"]
pub = {L: torch.tensor(ED[f"cols{L}"], device=dev).long() for L in (15, 19)}
brk = {L: torch.tensor(brake[f"L{L}"]["cols"], device=dev).long() for L in (10, 14)}

pred, full = torch.zeros_like(d), torch.zeros_like(d)
best = [[] for _ in jobs]
feed = {}
with torch.no_grad():
    for k in range(len(jobs)):
        for L2 in range(NL):  # what the MLP on floor L2 reads = stream after the floor, minus the floor's own MLP output
            full[k, L2] = read[L2] @ (rshift[k, L2 + 1] - down[L2].weight @ shift[k, L2]) / sd[k, L2]
    for L2 in range(1, NL):
        for L in range(L2):
            wire = read[L2] @ down[L].weight  # (j on L2, i on L)
            for k in range(len(jobs)):
                c = wire * shift[k, L][None, :] / sd[k, L2][:, None]
                pred[k, L2] += c.sum(1)
                if k == 0 and L in brk and L2 in pub:  # do the map's brake units feed the published columns?
                    feed[f"L{L}->L{L2}"] = dict(
                        from_brake_units=round(float(c[pub[L2]][:, brk[L]].sum(1).mean()), 4),
                        from_whole_floor=round(float(c[pub[L2]].sum(1).mean()), 4),
                        measured_d=round(float(d[0, L2][pub[L2]].mean()), 4))
                v, idx = (c.abs() * (lit[k, L2][:, None] & lit[k, L][None, :])).flatten().topk(TOP)
                keep = v > 0
                idx = idx[keep]
                j, i = idx // NU, idx % NU
                best[k] += [(float(c[jj, ii]), L, int(ii), L2, int(jj)) for jj, ii in zip(j.tolist(), i.tolist())]
wires = {}
for k, name in enumerate(names):
    best[k].sort(key=lambda t: -abs(t[0]))
    wires[name] = [[L, i, L2, j, round(c, 3)] for c, L, i, L2, j in best[k][:TOP]]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return round(float((a @ b) / (a.norm() * b.norm() + 1e-12)), 3)


check_sum = {name: dict(all_units=pearson(pred[k, 1:].flatten(), d[k, 1:].flatten()),
                        lit_units=pearson(pred[k, 1:][lit[k, 1:]], d[k, 1:][lit[k, 1:]]),
                        full_all_units=pearson(full[k, 1:].flatten(), d[k, 1:].flatten()),
                        full_lit_units=pearson(full[k, 1:][lit[k, 1:]], d[k, 1:][lit[k, 1:]]),
                        wires_vs_full=pearson(pred[k, 1:].flatten(), full[k, 1:].flatten()),
                        by_floor=[pearson(pred[k, L2], d[k, L2]) for L2 in range(1, NL)])
             for k, name in enumerate(names)}
for name, cs in check_sum.items():
    print(f"sum check  {name:24s} wires only: r(all) {cs['all_units']:+.3f} r(lit) {cs['lit_units']:+.3f} | "
          f"everything the unit reads: r(all) {cs['full_all_units']:+.3f} r(lit) {cs['full_lit_units']:+.3f} | wires vs everything {cs['wires_vs_full']:+.3f}")
print("brake -> published columns (loop job):", json.dumps(feed))

# ---- nudge check: is a drawn wire real? ---------------------------------------------------
state["mode"] = "nudge"
sample = {}
for name in names:
    for L, i, L2, j, c in wires[name][:NUDGE_TOP]:
        sample.setdefault((L, i), set()).add((L2, j))
test_ids = [ids_of(p) for p in problems[:: len(problems) // NUDGE_PROMPTS][:NUDGE_PROMPTS]]
predicted, measured = [], []
with torch.no_grad():
    base = []
    for ids in test_ids:
        state["nudge"] = None
        model(ids)
        base.append({L: v.clone() for L, v in state["seen"].items()})
    for (L, i), targets in sample.items():
        step = float(sd_pos[L, i])
        got = {t: 0.0 for t in targets}
        for ids, b in zip(test_ids, base):
            state["nudge"] = (L, i, step)
            model(ids)
            for (L2, j) in targets:
                got[(L2, j)] += float(state["seen"][L2][j] - b[L2][j]) / len(test_ids)
        for (L2, j) in targets:
            predicted.append(float(read[L2][j] @ down[L].weight[:, i]) * step)
            measured.append(got[(L2, j)])
state["nudge"] = None
P, M = np.array(predicted), np.array(measured)
check_nudge = dict(wires_tested=len(P), prompts=len(test_ids),
                   same_direction=int((np.sign(P) == np.sign(M)).sum()),
                   pearson_r=round(float(np.corrcoef(P, M)[0, 1]), 3),
                   median_measured_over_predicted=round(float(np.median(M / P)), 3),
                   quartiles_measured_over_predicted=[round(float(q), 3) for q in np.percentile(M / P, [25, 75])])
print("nudge check", json.dumps(check_nudge))

(HERE / "wires.json").write_text(json.dumps(dict(
    note="direct MLP-to-MLP path only; contribution = wire * job shift of source / job SD of target (Cohen's d units)",
    top=TOP, lit=LIT, positions=n, jobs=names, wires=wires,
    check_sum=check_sum, check_nudge=check_nudge, brake_feeds_published=feed)))
print(f"wires.json: {sum(len(w) for w in wires.values())} wires; strongest per job:",
      {name: wires[name][0] for name in names if wires[name]})
