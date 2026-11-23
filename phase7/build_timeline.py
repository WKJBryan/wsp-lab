"""Token timeline page: where each edited model departs from stock, closes thinking and ends.

Reads the saved MATH-500 generations (same 500 problems, same seed) of eighteen runs and packs
one row per problem per run into phase7/timeline.html, built from phase7/timeline_template.html.

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
    ("full", "Published edit + our repair at full dose", "broken", "phase3c/p7_stack_full"),
    ("s_mpro", "MiMo-pro alone: same row × 1.5", "solo", "phase3c/p4_mimo_pro"),
    ("s_gf50", "GLM-5.3-Flash alone, 50 tries allowed: row × 1.5, comma × 0.8", "solo", "phase3c/p10_fullbudget"),
    ("s_gf", "GLM-5.3-Flash alone: three rows rescaled", "solo", "phase3c/p4_glmflash"),
    ("s_glm", "GLM-5.3 alone: conditional boost + 5 format rows", "solo", "phase3c/p9_glm_single"),
    ("t_glm", "GLM-5.3 team of six: row × 1.35", "team", "phase3c/p9_glm_team"),
    ("t_mix", "Mixed team, four models: row × 1.35", "team", "phase3c/p9_mixed_team"),
    ("t_gf", "GLM-5.3-Flash team of six: gentle add-on to the row", "team", "phase3c/p9_glmflash_team"),
    ("t_live", "Surgeon + critic working live: uniform +0.02", "team", "phase3c/p9_interleaved"),
    ("t_mimo", "MiMo-flash team, short instructions", "team", "phase3c/p9_mimoflash_team_retry"),
]


def load(p):
    texts = json.load(gzip.open(ROOT / f"{p}_texts.json.gz", "rt", encoding="utf-8"))
    rescored = ROOT / f"{p}_rescored.json"
    rows = json.loads((rescored if rescored.exists() else ROOT / f"{p}.json").read_text(encoding="utf-8"))["rows"]
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

# self-check against the numbers already in the record (phase 3c / phase 8 / addendum 10)
want = dict(stock=6.0, orig=39.4, e1=27.8, c1=28.2, dout=26.4, din=5.8, map=26.2, half=37.0, full=4.4,
            s_mpro=27.8, s_gf50=27.4, s_gf=25.2, s_glm=23.0, t_glm=24.4, t_mix=23.6, t_gf=10.6, t_live=9.0, t_mimo=5.8)
for a in out:
    got = sum(x[4] for x in a["rows"]) / 5
    assert abs(got - want[a["key"]]) < 0.05, (a["key"], got)
assert all(x[0] == -1 for x in out[0]["rows"])

data = json.dumps(dict(arms=out, ids=ids, gold=[stock_r[i]["gold"] for i in ids]), separators=(",", ":"))
tpl = (ROOT / "phase7/timeline_template.html").read_text(encoding="utf-8")
assert tpl.count("/*DATA*/") == 1
(ROOT / "phase7/timeline.html").write_text(tpl.replace("/*DATA*/", data.replace("</", "<\\/")), encoding="utf-8")
print("wrote phase7/timeline.html", len(data) // 1024, "KB of data")
