"""Phase-7 atlas — a first label for every MLP unit (24 x 3584), from measurements already on disk.

Tags (a unit can carry several):
  first-word spike   |activation on token 0| >= 5 and >= 50x its usual size elsewhere   (positions.npz)
  loop marker        |d| >= 0.8, in-loop vs before-the-loop, while generating           (neurons.npz)
  subject: X         |d| >= 0.8 for subject X vs the other six while reading, with
                     prompt length regressed out                                        (questions_noprefix.npz)
  question length    |correlation of its per-question average with log prompt length| >= 0.5
  edited by authors  one of the 128 columns the published edit changes
  used in our repair one of the 128 loop units of the L10+L14 brake
No tag = "no label yet": nothing measured so far singles the unit out.
"Hard question" gets no tag: no unit reaches 0.8 once length is regressed out (Finding 22).

Outputs: phase7/labels.csv (one row per unit), phase7/labels_summary.json
Usage: uv run python phase7/label_units.py     (after scan_positions.py and probe_questions.py --no-prefix)
"""

import csv
import json
from collections import Counter

import numpy as np

from jobs import HERE, PO, job_stats, prompt_tokens, reading_masks, shift, without_length

LIT, NL, NU = 0.8, 24, 3584
P = np.load(PO / "positions.npz")
first, usual = P["first"], P["mean_abs"]
spike = (np.abs(first) >= 5) & (np.abs(first) >= 50 * usual)
loop_d = np.divide(*job_stats()[0][1:3])
Q = np.load(PO / "questions_noprefix.npz")["neurons"].astype(np.float32)
R = without_length(Q)
subjects = [(n, np.divide(*shift(R[a], R[b]))) for n, a, b in reading_masks() if n != "Hard question"]
logt = np.log(prompt_tokens())
z = (Q - Q.mean(0)) / (Q.std(0) + 1e-9)
r_len = np.tensordot((logt - logt.mean()) / logt.std(), z, axes=(0, 0)) / len(logt)  # (24, 3584)
ED = np.load(PO / "edit.npz")
brake = json.loads((HERE / "causal" / "brake_d100.json").read_text())["per_layer"]
edited = {(L, int(c)) for L in (15, 19) for c in ED[f"cols{L}"]}
repair = {(int(k[1:]), int(c)) for k, v in brake.items() for c in v["cols"]}

counts, primary_counts, per_floor, rows = Counter(), Counter(), {}, []
for L in range(NL):
    for u in range(NU):
        tags = []
        if spike[L, u]:
            tags.append("first-word spike")
        if abs(loop_d[L, u]) >= LIT:
            tags.append("loop marker")
        subj = sorted(((abs(d[L, u]), n, d[L, u]) for n, d in subjects if abs(d[L, u]) >= LIT), reverse=True)
        tags += [f"subject: {n}" for _, n, _ in subj]
        if abs(r_len[L, u]) >= 0.5:
            tags.append("question length")
        if (L, u) in edited:
            tags.append("edited by authors")
        if (L, u) in repair:
            tags.append("used in our repair")
        kinds = {t.split(":")[0] for t in tags}
        for k in kinds:
            counts[k] += 1
            per_floor.setdefault(k, [0] * NL)[L] += 1
        primary = tags[0] if tags else "no label yet"
        primary_counts[primary.split(":")[0]] += 1
        best = max(subjects, key=lambda s: abs(s[1][L, u]))
        rows.append([L, u, primary, "; ".join(tags), round(float(loop_d[L, u]), 2), best[0], round(float(best[1][L, u]), 2),
                     round(float(r_len[L, u]), 2), round(float(first[L, u]), 2), round(float(P["sd"][L, u]), 3)])
with open(HERE / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["floor", "unit", "primary_label", "all_tags", "loop_d", "strongest_subject", "subject_d_length_controlled",
                "corr_with_question_length", "first_token_activation", "usual_sd"])
    w.writerows(rows)
n_sub = Counter(sum(t.startswith("subject") for t in r[3].split("; ")) for r in rows)
summary = dict(units=NL * NU, tagged=NL * NU - primary_counts["no label yet"], no_label=primary_counts["no label yet"],
               units_with_tag=dict(counts), per_floor=per_floor,
               subjects_per_unit={str(k): v for k, v in sorted(n_sub.items())},
               each_subject={n: int((np.abs(d) >= LIT).sum()) for n, d in subjects})
(HERE / "labels_summary.json").write_text(json.dumps(summary, indent=1))
print(f"{summary['tagged']} of {summary['units']} units carry at least one tag ({summary['tagged'] / summary['units']:.1%}); {summary['no_label']} have no label yet")
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k:20s} {v:6d}   busiest floors {np.argsort(-np.array(per_floor[k]))[:3].tolist()}")
print("  subjects per unit:", summary["subjects_per_unit"])
print("  each subject:", summary["each_subject"])
both = sum(1 for r in rows if "loop marker" in r[3] and "subject" in r[3])
print(f"  loop marker AND a subject: {both} | loop marker AND question length: {sum(1 for r in rows if 'loop marker' in r[3] and 'question length' in r[3])}"
      f" | subject AND question length: {sum(1 for r in rows if 'subject' in r[3] and 'question length' in r[3])}")
