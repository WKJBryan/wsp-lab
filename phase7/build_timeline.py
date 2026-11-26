"""Token timeline page: where each edited model departs from stock, closes thinking and ends.

Reads the saved MATH-500 generations (same 500 problems, same seed) of twenty runs and packs
one row per problem per run into phase7/timeline.html, built from phase7/timeline_template.html.
The full-dose stacked arm is the clean rerun of 2026-11-24; the first run of that model was a
corrupted run and is not shown (ISSUES #20). The phase-14 and phase-15 arms are the blind edits
made under the original authors' rules by two editors of different strength (phase14/results.md,
phase15/results.md).

Usage: uv run python phase7/build_timeline.py
"""

import gzip
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained(ROOT / "phase3/models/stock")
ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

# key, plain name, group, files (relative to repo root, without suffix)
ARMS = [
    ("stock", "Stock model, no edit", "base", "phase3c/p3c_stock"),
    ("e1", "Blind edit by MiMo-flash alone (stop-token row × 1.5)", "skip", "phase3c/p4_mimo"),
    ("c1", "No edit, +9 on the stop token while generating", "skip", "phase8/c1_final"),
    ("dout", "Blind edit, output half only", "skip", "phase8/d_out"),
    ("din", "Blind edit, input half only", "none", "phase8/d_in"),
    ("orig", "Published edit (floors 15 + 19)", "change", "phase3c/p3c_c3_original"),
    ("map", "Our map repair (floors 10 + 14)", "change", "phase3c/p7_uperp_on_loop"),
    ("half", "Published edit + our repair at half dose", "change", "phase3c/p7_stack_half"),
    ("full", "Published edit + our repair at full dose (rerun of 24 Nov)", "change", "phase3c/p7_stack_full_v2"),
    ("s_mpro", "MiMo-pro alone: same row × 1.5", "solo", "phase3c/p4_mimo_pro"),
    ("s_gf50", "GLM-5.3-Flash alone, 50 tries allowed: row × 1.5, comma × 0.8", "solo", "phase3c/p10_fullbudget"),
    ("s_gf", "GLM-5.3-Flash alone: three rows rescaled", "solo", "phase3c/p4_glmflash"),
    ("s_glm", "GLM-5.3 alone: conditional boost + 5 format rows", "solo", "phase3c/p9_glm_single"),
    ("p14", "Blind edit by GLM-5.3-Flash under the original researchers' rules (one table in layer 0)", "solo", "phase3c/p14_final"),
    ("p15", "Blind edit by GLM-5.4 under the original researchers' rules (two switches in layer 14)", "solo", "phase3c/p15_final"),
    ("t_glm", "GLM-5.3 team of six: row × 1.35", "team", "phase3c/p9_glm_team"),
    ("t_mix", "Mixed team, four models: row × 1.35", "team", "phase3c/p9_mixed_team"),
    ("t_gf", "GLM-5.3-Flash team of six: gentle add-on to the row", "team", "phase3c/p9_glmflash_team"),
    ("t_live", "Surgeon + critic working live: uniform +0.02", "team", "phase3c/p9_interleaved"),
    ("t_mimo", "MiMo-flash team, short instructions", "team", "phase3c/p9_mimoflash_team_retry"),
]


def load(p):
    texts = json.load(gzip.open(ROOT / f"{p}_texts.json.gz", "rt", encoding="utf-8"))
    rescored = ROOT / f"{p}_rescored.json"
    data = json.loads((rescored if rescored.exists() else ROOT / f"{p}.json").read_text(encoding="utf-8"))
    # a run that failed a check, or was never given one, is not a measurement (ISSUES #20)
    v = data["summary"].get("validity") or {}
    problems = list(v.get("problems") or [])
    problems += [f"{p}: no {'fragment-tail' if k == 'soup_judged' else 'batch-position'} check on record"
                 for k in ("soup_judged", "admission_tested") if not v.get(k)]
    assert not problems, problems
    rows = data["rows"]
    assert len(rows) == 500 and sorted(r["id"] for r in rows) == sorted(texts), p
    return texts, {r["id"]: r for r in rows}


stock_t, stock_r = load(ARMS[0][3])
ids = list(stock_t)
assert len(ids) == 500
out = []
for key, name, group, p in ARMS:
    texts, rows = load(p)
    assert set(texts) == set(ids), key
    packed = []
    for i in ids:
        a, b, r = stock_t[i], texts[i], rows[i]
        assert r["closed"] == ("</think>" in b), (key, i)   # rows and generations are the same run
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        split = -1 if a == b else ntok(a[:n])  # tokens shared with stock before the first difference
        close = -1
        if r["closed"]:
            close = r.get("think_end_at")
            if close is None:  # phase-3c rows do not store it
                assert "</think>" in b, (key, i)
                close = ntok(b.split("</think>")[0])
        ok = r["ok_rescored"] if "ok_rescored" in r else r["ok"]
        packed.append([split, close, r["n_tokens"], int(r["finish"] == "stop"), int(bool(ok))])
    acc = sum(x[4] for x in packed) / 5
    print(f"{key:6s} acc {acc:5.1f}%  closed {sum(x[1] >= 0 for x in packed)}  finished {sum(x[3] for x in packed)}")
    out.append(dict(key=key, name=name, group=group, rows=packed))

# self-check against the numbers in the record (phase 3c / phase 8 / phase 14 / addendum 10), as
# re-scored with the final answer checker (ISSUES #19, #30, #31)
want = dict(stock=9.4, orig=54.2, e1=42.2, c1=44.8, dout=41.0, din=10.0, map=36.4, half=50.8, full=52.2,
            s_mpro=40.6, s_gf50=41.6, s_gf=37.2, s_glm=31.8, p14=11.0, p15=53.0, t_glm=35.2, t_mix=33.2, t_gf=17.0, t_live=13.0,
            t_mimo=9.2)
for a in out:
    got = sum(x[4] for x in a["rows"]) / 5
    assert abs(got - want[a["key"]]) < 0.05, (a["key"], got)
assert all(x[0] == -1 for x in out[0]["rows"])

data = json.dumps(dict(arms=out, ids=ids, gold=[stock_r[i]["gold"] for i in ids]), separators=(",", ":"))
tpl = (ROOT / "phase7/timeline_template.html").read_text(encoding="utf-8")
assert tpl.count("/*DATA*/") == 1
(ROOT / "phase7/timeline.html").write_text(tpl.replace("/*DATA*/", data.replace("</", "<\\/")), encoding="utf-8")
print("wrote phase7/timeline.html", len(data) // 1024, "KB of data")
