"""Phase-7 atlas — do the reading-job lights survive without the fixed opening tokens? (protocol addendum 7)

Compares, per reading job, Cohen's d of every MLP unit computed three ways:
  before   averages over every prompt position                     (probe_out/questions.npz)
  after    averages leave out the 3 opening template tokens         (probe_out/questions_noprefix.npz)
  length   "after", with prompt length (linear + log, the part-3 control) regressed out of every unit
           first — conservative: difficulty and length really go together, so this removes real signal too

Outputs: phase7/reading_remeasure.json
Usage: uv run python phase7/remeasure_reading.py     (after probe_questions.py --no-prefix)
"""

import json
from pathlib import Path

import numpy as np
from jobs import PO, prompt_tokens, reading_masks, shift, without_length

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIT = 0.8

old = np.load(PO / "questions.npz")["neurons"].astype(np.float32)
new = np.load(PO / "questions_noprefix.npz")["neurons"].astype(np.float32)
n_tok = prompt_tokens()
resid = without_length(new)

out, total = {}, dict(before=0, after=0, stay=0, length=0, stay_length=0)
for name, a, b in reading_masks():
    d0, d1, d2 = (np.divide(*shift(x[a], x[b])) for x in (old, new, resid))
    on0, on1, on2 = (np.abs(d) >= LIT for d in (d0, d1, d2))
    row = dict(lit_before=int(on0.sum()), lit_after=int(on1.sum()), stay_lit=int((on0 & on1).sum()), newly_lit=int((~on0 & on1).sum()),
               r_before_after=round(float(np.corrcoef(d0.ravel(), d1.ravel())[0, 1]), 4),
               lit_length_regressed=int(on2.sum()), stay_lit_length_regressed=int((on0 & on2).sum()),
               after_stay_length_regressed=int((on1 & on2).sum()),  # of the lights the page now shows
               lit05_after=int((np.abs(d1) >= 0.5).sum()), lit05_length_regressed=int((np.abs(d2) >= 0.5).sum()),
               max_abs_d_after=round(float(np.abs(d1).max()), 3), max_abs_d_length_regressed=round(float(np.abs(d2).max()), 3),
               median_kept_by_lit_units=round(float(np.median(np.abs(d2[on1]) / np.abs(d1[on1]))), 3),  # compare with shrink_if_job_only
               # what regressing length would cost a unit that tracks the JOB and not length: d shrinks by this factor
               shrink_if_job_only=round(float(1 - np.corrcoef(a[a | b].astype(float), np.log(n_tok[a | b]))[0, 1] ** 2), 3),
               L14_u46=[round(float(d[14, 46]), 3) for d in (d0, d1, d2)],
               biggest_drops=[[int(L), int(u), round(float(d0[L, u]), 2), round(float(d1[L, u]), 2)]
                              for L, u in zip(*np.unravel_index(np.argsort(-(np.abs(d0) - np.abs(d1)).ravel())[:5], d0.shape))])
    out[name] = row
    for k, v in (("before", "lit_before"), ("after", "lit_after"), ("stay", "stay_lit"), ("length", "lit_length_regressed"), ("stay_length", "stay_lit_length_regressed")):
        total[k] += row[v]
    print(f"{name:24s} lights {row['lit_before']:6d} -> {row['lit_after']:6d} | stay lit {row['stay_lit'] / max(1, row['lit_before']):6.1%} | r {row['r_before_after']:.4f} | "
          f"length-regressed: {row['lit_length_regressed']:6d} lit, {row['stay_lit_length_regressed'] / max(1, row['lit_before']):6.1%} of old stay, "
          f"{row['after_stay_length_regressed']} of the new stay, {row['lit05_length_regressed']} at 0.5, max |d| {row['max_abs_d_length_regressed']}, "
          f"lit units keep a median {row['median_kept_by_lit_units']:.0%} of their d (a job-only unit would keep {row['shrink_if_job_only']:.0%}) | "
          f"at 0.5: {row['lit05_after']} -> {row['lit05_length_regressed']} | max |d| {row['max_abs_d_after']} -> {row['max_abs_d_length_regressed']}")
print("all reading jobs:", total, f"| stay {total['stay'] / total['before']:.1%} | stay after length regression {total['stay_length'] / total['before']:.1%}")
print("biggest drops (floor, unit, d before, d after):", {n: r["biggest_drops"] for n, r in out.items()})
(HERE / "reading_remeasure.json").write_text(json.dumps(dict(lit=LIT, jobs=out, total=total), indent=1))
