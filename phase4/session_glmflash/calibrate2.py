"""Offline calibration combos: </think> scale x comma suppression x im_end boost.
Logs everything to file. No eval budget used."""
import os, sys, json, re, torch
from fractions import Fraction
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
TOK_CLOSE, IM_END, EOT = 248069, 248046, 248044
COMMA = 11
N_PROB = 4
OUT = os.path.expanduser("~/wsp4/sandbox/calib2_log.txt")
LOG = open(OUT, "w")

def log(*a):
    print(*a); LOG.write(" ".join(str(x) for x in a) + "\n"); LOG.flush()

tok = AutoTokenizer.from_pretrained(MODEL)
probs = json.load(open(os.path.expanduser("~/wsp4/sandbox/practice.json")))[:N_PROB]

# ---- exact scorer from eval_pass.py ----
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

print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval().to("cuda")
W = model.get_input_embeddings().weight
base_close = W[TOK_CLOSE].detach().clone()
base_comma = W[COMMA].detach().clone()
base_imend = W[IM_END].detach().clone()

prompts = [tok.apply_chat_template([[{"role": "user", "content": p["q"]}]],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True) for p in probs]

COMBOS = [
    dict(name="c1.35_comma.50_end1.15", close=1.35, comma=0.50, imend=1.15),
    dict(name="c1.55_comma.50_end1.15", close=1.55, comma=0.50, imend=1.15),
]

def set_rows(close, comma, imend):
    with torch.no_grad():
        W[TOK_CLOSE] = base_close * close
        W[COMMA] = base_comma * comma
        W[IM_END] = base_imend * imend

for combo in COMBOS:
    set_rows(combo["close"], combo["comma"], combo["imend"])
    ncorr, nclosed, nstop = 0, 0, 0
    log("=" * 90)
    log("COMBO", combo["name"])
    for pi, (p, pr) in enumerate(zip(probs, prompts)):
        torch.manual_seed(1234 + pi)
        ids = tok(pr, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, eos_token_id=[IM_END, EOT], pad_token_id=EOT,
                                 max_new_tokens=1200, do_sample=True, temperature=0.6,
                                 top_p=0.95, top_k=20, min_new_tokens=8)
        gen = out[0][ids["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        closed = "</think>" in text
        stopped = (gen[-1].item() in (IM_END, EOT)) if len(gen) else False
        pred = extract(text)
        ok = match(p["gold"], pred)
        ncommas = text.count(",")
        ncorr += ok; nclosed += closed; nstop += stopped
        log("  %-9s gold=%-8s pred=%-10s ok=%s closed=%s stop=%s ntok=%4d commas=%d" % (
            p["id"], p["gold"], str(pred), ok, closed, stopped, len(gen), ncommas))
        if closed:
            post = text.split("</think>", 1)[1]
            log("    POST-TAIL: %r" % post[-220:].replace("\n", "\\n"))
    log("  SUMMARY %s: acc=%d/%d closed=%d/%d stopped=%d/%d" % (
        combo["name"], ncorr, N_PROB, nclosed, N_PROB, nstop, N_PROB))

set_rows(1.0, 1.0, 1.0)
log("rows restored")
LOG.close()
