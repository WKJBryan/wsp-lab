"""Phase-7 knockout test of the subject labels (protocol addendum 9). Forward passes only, stock model.

Even-index problems choose the units; odd-index problems are the test.
  knockout set, per subject   the 1,000 units with the highest (d x sign of average activation) for
                              "this subject vs the other six" while reading, prompt length regressed out
  control, per subject        the same number of units on each floor, random, outside the knockout set
  readout                     the stock model's saved answers to the odd problems, first 300 generated
                              tokens, teacher-forced: extra next-token loss per subject of the question
  selectivity                 extra loss on the knocked-out subject's text / mean over the other six

Outputs: phase7/knockout.json
Usage: uv run python phase7/knockout_subjects.py
"""

import gzip
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jobs import HERE, PO, reading_masks, shift, without_length

ROOT = HERE.parent
NL, NU, K, MAXGEN, dev = 24, 3584, 1000, 300, "cuda"

# ---- choose the units on the even half ----
Q = np.load(PO / "questions_noprefix.npz")["neurons"].astype(np.float32)
R = without_length(Q)
fit = np.arange(500) % 2 == 0
masks = [(n, a, b) for n, a, b in reading_masks() if n != "Hard question"]
names = [n for n, _, _ in masks]
sign = np.sign(Q[fit].mean(0))
rng = np.random.default_rng(0)
sets, ctrls = {}, {}
for n, a, b in masks:
    score = (np.divide(*shift(R[a & fit], R[b & fit])) * sign).ravel()
    ko = np.zeros(NL * NU, bool)
    ko[np.argsort(-score)[:K]] = True
    ko = ko.reshape(NL, NU)
    ct = np.zeros((NL, NU), bool)
    for L in range(NL):
        ct[L, rng.choice(np.flatnonzero(~ko[L]), int(ko[L].sum()), replace=False)] = True
    sets[n], ctrls[n] = ko, ct

# ---- the test text: odd problems ----
stock = ROOT / "phase3" / "models" / "stock"
tok = AutoTokenizer.from_pretrained(stock)
model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.bfloat16).eval().to(dev)
down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
        if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, NU)}
assert sorted(down) == list(range(NL))
state = dict(mask=None)


def hook(L):
    def fn(_, inp):
        if state["mask"] is not None and bool(state["mask"][L].any()):
            a = inp[0].clone()
            a[..., state["mask"][L]] = 0
            return (a,)
    return fn


for L, m in down.items():
    m.register_forward_pre_hook(hook(L))

problems = json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))
texts = json.loads(gzip.open(ROOT / "phase3c" / "p3c_stock_texts.json.gz", "rt", encoding="utf-8").read())
subject_of = np.full(500, -1)
for k, (_, a, _) in enumerate(masks):
    subject_of[a] = k
assert (subject_of >= 0).all()
seqs = []
for i in np.flatnonzero(~fit):
    p = problems[i]
    prompt = tok.apply_chat_template([{"role": "user", "content": p["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    g_ids = tok(texts[p["id"]], add_special_tokens=False, return_tensors="pt").input_ids[:, :MAXGEN]
    if g_ids.shape[1] >= 50:
        seqs.append((torch.cat([p_ids, g_ids], 1).to(dev), p_ids.shape[1], int(subject_of[i])))
n_tok = np.zeros(len(names))
for ids, plen, s in seqs:
    n_tok[s] += ids.shape[1] - plen


def run(mask):
    """summed next-token loss on the generated tokens of every test problem"""
    state["mask"] = None if mask is None else torch.from_numpy(mask).to(dev)
    with torch.no_grad():
        return np.array([float(F.cross_entropy(model(ids).logits[0, plen - 1:-1].float(), ids[0, plen:], reduction="sum")) for ids, plen, _ in seqs])


subj = np.array([s for _, _, s in seqs])
toks = np.array([ids.shape[1] - plen for ids, plen, _ in seqs], float)
by_subject = lambda per_seq, idx=None: np.array([(per_seq[subj == k] if idx is None else per_seq[idx[k]]).sum()
                                                  / (toks[subj == k] if idx is None else toks[idx[k]]).sum() for k in range(len(names))])
base_seq = run(None)
base = by_subject(base_seq)
print(f"test problems {len(seqs)}; tokens per subject {dict(zip(names, n_tok.astype(int).tolist()))}", flush=True)
print("unpruned loss per subject:", dict(zip(names, np.round(base, 3).tolist())), flush=True)
per_seq = {kind: [run(d[n]) - base_seq for n in names] for kind, d in (("knockout", sets), ("control", ctrls))}  # extra loss per test problem
extra = {kind: np.stack([by_subject(x) for x in v]) for kind, v in per_seq.items()}  # [knocked-out subject, text subject]


def selectivity(M):
    return [float(M[k, k] / np.delete(M[k], k).mean()) for k in range(len(names))]


sel = {kind: selectivity(M) for kind, M in extra.items()}
# POST-HOC (not in the addendum): how sure is each selectivity? resample the test problems within each subject
boot = {kind: [] for kind in extra}
brng = np.random.default_rng(1)
for _ in range(2000):
    idx = [brng.choice(np.flatnonzero(subj == k), int((subj == k).sum())) for k in range(len(names))]
    for kind in extra:
        boot[kind].append(selectivity(np.stack([by_subject(x, idx) for x in per_seq[kind]])))
boot = {kind: np.array(v) for kind, v in boot.items()}  # (2000, 7)
ci = {kind: np.percentile(v, [5, 95], axis=0).T for kind, v in boot.items()}
p_above_1 = (boot["knockout"] > 1).mean(0)
p_above_ctrl = (boot["knockout"] > boot["control"]).mean(0)
own_is_worst = [int(np.argmax(extra["knockout"][:, k]) == k) for k in range(len(names))]
for k, n in enumerate(names):
    print(f"{n:24s} knockout: own +{extra['knockout'][k, k]:.3f} others +{np.delete(extra['knockout'][k], k).mean():.3f} selectivity {sel['knockout'][k]:.2f} | "
          f"control: own +{extra['control'][k, k]:.3f} others +{np.delete(extra['control'][k], k).mean():.3f} selectivity {sel['control'][k]:.2f} | "
          f"own knockout is the worst for its own text: {bool(own_is_worst[k])} | floors {np.argsort(-sets[n].sum(1))[:3].tolist()}", flush=True)
    print(f"{'':24s} 90% range: knockout {ci['knockout'][k][0]:.2f}-{ci['knockout'][k][1]:.2f}, control {ci['control'][k][0]:.2f}-{ci['control'][k][1]:.2f} | "
          f"chance selectivity > 1: {p_above_1[k]:.0%} | chance knockout more selective than control: {p_above_ctrl[k]:.0%}", flush=True)
print("extra loss, rows = knocked-out subject, columns = text subject:\n", np.round(extra["knockout"], 3))
print("median selectivity: knockout", round(float(np.median(sel["knockout"])), 2), "| control", round(float(np.median(sel["control"])), 2))
overlap = np.array([[int((sets[a] & sets[b]).sum()) for b in names] for a in names])
(HERE / "knockout.json").write_text(json.dumps(dict(
    subjects=names, units_per_set=K, test_problems=len(seqs), tokens_per_subject=n_tok.astype(int).tolist(), unpruned_loss=np.round(base, 4).tolist(),
    extra_loss={k: np.round(M, 4).tolist() for k, M in extra.items()}, selectivity={k: [round(x, 3) for x in v] for k, v in sel.items()},
    own_knockout_is_worst=own_is_worst, set_overlap=overlap.tolist(),
    posthoc_bootstrap=dict(selectivity_90pct={k: np.round(v, 3).tolist() for k, v in ci.items()},
                           chance_selectivity_above_1=np.round(p_above_1, 3).tolist(), chance_above_control=np.round(p_above_ctrl, 3).tolist()),
    units_per_floor={n: sets[n].sum(1).tolist() for n in names}), indent=1))
