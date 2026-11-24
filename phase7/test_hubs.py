"""Phase-7 causal check on the hub units of Finding 20 (protocol addendum 6, parts A and B).

Forward passes only, stock model (bf16, as in the part-2 and part-3 probes). A unit is "clamped"
by overwriting its activation (the input of down_proj) with one constant at every position, so
it no longer differs between conditions.

  A. reading   re-read the 500 MATH-500 prompts with the hub held at its all-position mean;
               recompute Cohen's d of every unit for the jobs where it is a hub
  B. loop      replay the 80 stuck generations of sets.json (teacher-forced) with the hub held
               at its before-the-loop mean; recompute the loop-vs-pre d
  For both: wire prediction for a target = -(its contribution in wires.json); control = the same
  clamp on the same-floor unit with the closest activation SD that is the source of no shipped wire.

Outputs: phase7/hubs.json
Usage: uv run python phase7/test_hubs.py
"""

import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jobs import reading_masks, shift

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
READ_HUBS = {(14, 46): ["Hard question", "Number Theory", "Geometry", "Precalculus"], (5, 1087): ["Counting & Probability"]}
LOOP_HUB = (19, 11)
LIT = 0.8
dev = "cuda"
T0 = time.time()

W = json.loads((HERE / "wires.json").read_text())["wires"]
sources = {(w[0], w[1]) for ws in W.values() for w in ws}
stock = ROOT / "phase3" / "models" / "stock"
tok = AutoTokenizer.from_pretrained(stock)
model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.bfloat16).eval().to(dev)
down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
        if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, 3584)}
NL, NU = 24, 3584
assert sorted(down) == list(range(NL))
state = dict(clamp=None, pos=None)  # clamp: (layer, unit, value); pos: positions to keep, or None for whole-prompt sums
cur = {}


def hook(L):
    def fn(_, inp):
        a = inp[0]
        if state["clamp"] and state["clamp"][0] == L:
            a = a.clone()
            a[..., state["clamp"][1]] = state["clamp"][2]
        x = a[0].float()  # only small summaries are kept, never the whole (T, 3584) activation
        cur[L] = (x.sum(0), x.pow(2).sum(0)) if state["pos"] is None else x[state["pos"]]
        return (a,)
    return fn


for L, m in down.items():
    m.register_forward_pre_hook(hook(L))

problems = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))
qs = {p["id"]: p["q"] for p in problems}
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
masks = reading_masks()


def read_pass(label, clamp=None):
    """d per reading job {name: (24, 3584)}, plus per-position mean and SD of every unit"""
    state.update(clamp=clamp, pos=None)
    Q = np.zeros((len(problems), NL, NU), np.float32)
    s1, s2, n = torch.zeros(NL, NU, dtype=torch.float64, device=dev), torch.zeros(NL, NU, dtype=torch.float64, device=dev), 0
    with torch.no_grad():
        for k, p in enumerate(problems):
            ids = tok(chat(p["q"]), return_tensors="pt").input_ids.to(dev)
            model(ids)
            a, b = torch.stack([cur[L][0] for L in range(NL)]), torch.stack([cur[L][1] for L in range(NL)])
            Q[k] = (a / ids.shape[1]).cpu().numpy()
            s1 += a.double(); s2 += b.double(); n += ids.shape[1]
    mean = s1 / n
    print(f"[{time.time() - T0:5.0f}s] reading pass done: {label}", flush=True)
    return ({name: np.divide(*shift(Q[a], Q[b])) for name, a, b in masks},
            mean.cpu().numpy(), (s2 / n - mean ** 2).clamp_min(1e-12).sqrt().cpu().numpy())


texts = json.loads(gzip.open(ROOT / "phase3c" / "p3c_stock_texts.json.gz", "rt", encoding="utf-8").read())
stuck = [it for it in json.loads((HERE / "sets.json").read_text())["items"] if it["group"] == "stuck"]
assert len(stuck) == 80 and all({"loop", "pre"} <= set(it["positions"]) for it in stuck)


def loop_pass(label, clamp=None):
    """loop-vs-pre d (24, 3584) over the 80 stuck generations, plus the before-the-loop mean and the pooled SD"""
    acc = {r: [torch.zeros(NL, NU, dtype=torch.float64, device=dev), torch.zeros(NL, NU, dtype=torch.float64, device=dev), 0] for r in ("loop", "pre")}
    with torch.no_grad():
        for it in stuck:
            p_ids = tok(chat(qs[it["id"]]), return_tensors="pt").input_ids
            g_ids = tok(texts[it["id"]], add_special_tokens=False, return_tensors="pt").input_ids[:, :it["n_ctx"]]
            nl = len(it["positions"]["loop"])
            state.update(clamp=clamp, pos=torch.tensor([p_ids.shape[1] + p for r in ("loop", "pre") for p in it["positions"][r]], device=dev))
            model(torch.cat([p_ids, g_ids], 1).to(dev))
            x = torch.stack([cur[L] for L in range(NL)]).double()  # (24, npos, 3584)
            for r, part in (("loop", x[:, :nl]), ("pre", x[:, nl:])):
                acc[r][0] += part.sum(1); acc[r][1] += part.pow(2).sum(1); acc[r][2] += part.shape[1]
    m = {r: a[0] / a[2] for r, a in acc.items()}
    v = {r: (acc[r][1] / acc[r][2] - m[r] ** 2).clamp_min(1e-12) for r in acc}
    sd = ((v["loop"] + v["pre"]) / 2).sqrt()
    print(f"[{time.time() - T0:5.0f}s] loop replay done: {label}", flush=True)
    return ((m["loop"] - m["pre"]) / sd).cpu().numpy(), m["pre"].cpu().numpy(), sd.cpu().numpy()


def control_of(hub, sd):
    L, u = hub
    cand = [c for c in range(NU) if (L, c) not in sources and c != u]
    return L, min(cand, key=lambda c: abs(sd[L, c] - sd[L, u]))


def pattern(d0, d1, L):
    """how much a job's lights on the floors above L changed"""
    a, b = d0[L + 1:].ravel(), d1[L + 1:].ravel()
    on0, on1 = np.abs(a) >= LIT, np.abs(b) >= LIT
    return dict(pattern_r=round(float(np.corrcoef(a, b)[0, 1]), 4), lit_before=int(on0.sum()), lit_after=int(on1.sum()),
                went_dark=int((on0 & ~on1).sum()), newly_lit=int((~on0 & on1).sum()), mean_abs_change=round(float(np.abs(b - a).mean()), 4))


def targets(d0, d1, hub, job):
    """the hub's shipped wires for this job: predicted change -c vs the measured change of each target's d"""
    tg = [(w[2], w[3], w[4]) for w in W[job] if (w[0], w[1]) == hub]
    pred = np.array([-c for _, _, c in tg])
    meas = np.array([d1[L2, j] - d0[L2, j] for L2, j, _ in tg])
    return dict(targets=len(tg), same_direction=int((np.sign(pred) == np.sign(meas)).sum()),
                median_measured_over_predicted=round(float(np.median(meas / pred)), 3),
                quartiles=[round(float(q), 3) for q in np.percentile(meas / pred, [25, 75])],
                pearson_r=round(float(np.corrcoef(pred, meas)[0, 1]), 3),
                mean_abs_predicted=round(float(np.abs(pred).mean()), 3), mean_abs_measured=round(float(np.abs(meas).mean()), 3))


def report(out, hub, ctrl, job, d0, dh, dc, sd, clamp_value):
    res = dict(hub=list(hub), control=list(ctrl), clamp_value=round(clamp_value, 4), sd_hub=round(float(sd[hub]), 4), sd_control=round(float(sd[ctrl]), 4),
               d_hub=round(float(d0[hub]), 3), d_hub_clamped=round(float(dh[hub]), 3),
               wires=targets(d0, dh, hub, job), hub_clamped=pattern(d0, dh, hub[0]), control_clamped=pattern(d0, dc, hub[0]))
    out[f"L{hub[0]}u{hub[1]} | {job}"] = res
    print(f"L{hub[0]} u{hub[1]} | {job}:", json.dumps(res), flush=True)
    (HERE / "hubs.json").write_text(json.dumps(out, indent=1))


out = {}
# ---- A. reading ----
d0, mean_pos, sd_pos = read_pass("baseline")
print("baseline lights on at 0.8 (page: Hard question 289, Counting & Probability 14,777):",
      {n: int((np.abs(d0[n]) >= LIT).sum()) for n in ("Hard question", "Counting & Probability")}, flush=True)
for hub, jobs in READ_HUBS.items():
    ctrl = control_of(hub, sd_pos)
    dh = read_pass(f"hub L{hub[0]} u{hub[1]}", (*hub, float(mean_pos[hub])))[0]
    dc = read_pass(f"control L{ctrl[0]} u{ctrl[1]}", (*ctrl, float(mean_pos[ctrl])))[0]
    for job in jobs:
        report(out, hub, ctrl, job, d0[job], dh[job], dc[job], sd_pos, float(mean_pos[hub]))

# ---- B. loop replay ----
l0, pre_mean, sd_loop = loop_pass("baseline")
print("baseline loop lights on at 0.8 (page: 3,682):", int((np.abs(l0) >= LIT).sum()), flush=True)
ctrl = control_of(LOOP_HUB, sd_loop)
lh = loop_pass(f"hub L{LOOP_HUB[0]} u{LOOP_HUB[1]}", (*LOOP_HUB, float(pre_mean[LOOP_HUB])))[0]
lc = loop_pass(f"control L{ctrl[0]} u{ctrl[1]}", (*ctrl, float(pre_mean[ctrl])))[0]
report(out, LOOP_HUB, ctrl, "Stuck in a loop", l0, lh, lc, sd_loop, float(pre_mean[LOOP_HUB]))
print("ALL DONE", flush=True)
