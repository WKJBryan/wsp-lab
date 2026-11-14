import json, re
import torch
from fractions import Fraction

# --- scorer copied verbatim from eval_pass.py ---
MARKER_RE = re.compile(
    r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should", re.I)
STAR_VIEW = str.maketrans({"*": " "})

def canonical(s):
    s = s.strip().replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    return s[1:] if s.startswith("+") else s

def to_number(s):
    s = canonical(s)
    try:
        if re.fullmatch(r"-?\d+", s): return Fraction(int(s))
        if re.fullmatch(r"-?\d*\.\d+", s): return Fraction(s)
        m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
        if m and int(m.group(2)): return Fraction(int(m.group(1)), int(m.group(2)))
        m = re.fullmatch(r"\\frac\{(-?[^{}]+)\}\{(-?[^{}]+)\}", s)
        if m:
            a, b = to_number(m.group(1)), to_number(m.group(2))
            if a is not None and b is not None and b != 0: return a / b
    except Exception:
        return None
    return None

def match(gold, pred):
    if pred is None: return False
    g, p = canonical(gold), canonical(pred)
    if g == p: return True
    gn, pn = to_number(g), to_number(p)
    return gn is not None and pn is not None and gn == pn

def clean_inner(seg):
    seg = seg.strip().strip("$").strip()
    if not seg: return None
    maths = re.findall(r"\$([^$]+)\$", seg)
    if maths: return canonical(maths[-1])
    if "\\" in seg: return canonical(seg)
    if re.search("[a-zA-Z]", seg):
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
        if nums: return nums[-1]
        words = re.findall(r"[A-Za-z]+", seg)
        return canonical(words[-1]) if words else None
    return canonical(seg)

def clean(seg):
    seg = seg.strip()
    seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
    seg = seg.replace("\\[", "").replace("\\]", "")
    if "=" in seg: seg = seg.split("=")[-1]
    seg = seg.strip().strip("*").strip().rstrip(".").strip()
    bolds = re.findall(r"\*\*([^*]+)\*\*", seg)
    return clean_inner(bolds[-1]) if bolds else clean_inner(seg)

def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed: return boxed[-1]
    if "</think>" not in text: return None
    answer = text.split("</think>", 1)[1]
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if not lines: return None
    best = None
    for i, l in enumerate(lines):
        m = MARKER_RE.search(l.lower().translate(STAR_VIEW))
        if m:
            payload = l[m.end():].lstrip(":").strip()
            if not payload and i + 1 < len(lines):
                payload = lines[i + 1]
            c = clean(payload)
            if c: best = c
    return best if best else clean(lines[-1])
# --- end verbatim scorer ---

gold = {p["id"]: p["gold"] for p in json.load(open("/home/user/wsp4/sandbox/practice.json"))}

def load(tag):
    d = torch.load(f"/home/user/wsp4/sandbox/replay_stage3_{tag}.pt", weights_only=False)
    out = {}
    for r in d["results"]:
        ans = extract(r["gen_text"])
        out[r["id"]] = dict(fire=r["fire"], spam=r["spam"], finish=r["finish"],
                            correct=match(gold[r["id"]], ans), answer=ans)
    return out

ctrl = load("ctrl02")
seeds = {"a03_s0": "seed20261105", "a03_s1": "seed90210"}
n_degrade_total = 0
for tag, label in seeds.items():
    arm = load(tag)
    degr = []
    for i in ctrl:
        c, a = ctrl[i], arm[i]
        lost_answer = (c["finish"] == "imend") and (a["finish"] != "imend")
        lost_correct = c["correct"] and (not a["correct"])
        if lost_answer or lost_correct:
            degr.append((i, "imend->" + a["finish"], c["correct"], a["correct"], c["answer"], a["answer"]))
    sp3 = [i for i in arm if arm[i]["spam"] >= 3]
    fires_c = [ctrl[i]["fire"] for i in ctrl]
    fires_a = [arm[i]["fire"] for i in ctrl]
    delta = [(b - a) for a, b in zip(fires_c, fires_a) if a and b]
    print(f"== {label}: degraded={len(degr)} (gate <=2) spam>=3 rows={len(sp3)} (gate 0)", flush=True)
    print("   fire median ctrl %.0f -> a03 %.0f ; mean delta %.0f (n=%d both-fired)" % (
        sorted([f for f in fires_c if f])[len([f for f in fires_c if f])//2], sorted([f for f in fires_a if f])[len([f for f in fires_a if f])//2],
        sum(delta)/max(len(delta),1), len(delta)), flush=True)
    for d in degr:
        print("   DEGRADE", d[0], d[1], "correct", d[2], "->", d[3])
        print("     ctrl_ans:", repr(d[4]), "-> a03_ans:", repr(d[5]), flush=True)
    n_degrade_total += len(degr)
print("TOTAL degraded across both seeds:", n_degrade_total, flush=True)
