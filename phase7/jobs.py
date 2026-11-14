"""Phase-7 atlas — the nine "jobs" of the light map, as (name, mean shift, pooled SD) per MLP unit,
plus the same mean shift of the residual stream at every layer boundary.

Cohen's d of a unit for a job is shift / SD. Shared by build_lightmap.py, build_wires.py, label_units.py.
The reading jobs have prompt length (linear + log) regressed out first: longer questions move many units on
their own, and "Hard question" was mostly that (Finding 22).

Jobs and where their numbers come from (all stock model, probe_out/*.npz):
  Stuck in a loop   in-loop vs before-the-loop positions, while GENERATING   (neurons.npz)
  7 MATH subjects   that subject vs the other six, while READING the question (questions_noprefix.npz)
  Hard question     level 4-5 vs level 1-2, while READING the question        (questions_noprefix.npz)
"""

from pathlib import Path

import numpy as np
from datasets import load_from_disk

HERE = Path(__file__).resolve().parent
PO = HERE / "probe_out"


def shift(a, b):
    """(mean difference, pooled SD) per unit; a, b: (items, 24, 3584)"""
    return a.mean(0) - b.mean(0), np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)


def reading_masks():
    """[(name, in-group mask, comparison mask)] over the 500 MATH-500 problems, in job order"""
    ds = load_from_disk(str(HERE.parent / "phase3" / "data" / "math500"))
    subject, level = np.array(ds["subject"]), np.array(ds["level"])
    return ([(str(s), subject == s, subject != s) for s in sorted(set(subject))]
            + [("Hard question", level >= 4, level <= 2)])


def prompt_tokens():
    """token count of each of the 500 questions (question text only, as in the part-3 length control)"""
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE.parent / "phase3" / "models" / "stock")
    qs = json.loads((HERE.parent / "phase3c" / "problems.json").read_text(encoding="utf-8"))
    return np.array([len(tok(p["q"]).input_ids) for p in qs], float)


def without_length(Q):
    """Q (500, 24, 3584) with prompt length (linear + log) regressed out of every unit"""
    n = prompt_tokens()
    A = np.stack([np.ones(len(n)), n, np.log(n)], 1)
    flat = Q.reshape(len(n), -1).astype(np.float64)
    return (flat - A @ np.linalg.lstsq(A, flat, rcond=None)[0]).reshape(Q.shape).astype(np.float32)


def job_stats():
    """[(name, shift (24, 3584), sd (24, 3584), resid_shift (25, 1024))]"""
    NZ = np.load(PO / "neurons.npz")

    def gen_stats(region):
        n = sum(int(NZ[f"cnt_{region}_{h}"]) for h in (0, 1))
        m = sum(NZ[f"sum_{region}_{h}"] for h in (0, 1)) / n
        return m, np.maximum(sum(NZ[f"sq_{region}_{h}"] for h in (0, 1)) / n - m * m, 1e-12)

    (ml, vl), (mp, vp) = gen_stats("loop"), gen_stats("pre")
    R = np.load(PO / "resid.npz")
    jobs = [("Stuck in a loop", ml - mp, np.sqrt((vl + vp) / 2),
             R["resid_loop"].astype(np.float64).mean(0) - R["resid_pre"].astype(np.float64).mean(0))]
    QZ = np.load(PO / "questions_noprefix.npz")  # averages leave out the 3 opening tokens every prompt shares (addendum 7)
    Q, H = without_length(QZ["neurons"].astype(np.float32)), without_length(QZ["mean"]).astype(np.float64)
    for name, a, b in reading_masks():
        jobs.append((name, *shift(Q[a], Q[b]), H[a].mean(0) - H[b].mean(0)))
    return jobs
