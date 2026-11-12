"""Offline calibration sweep: scale </think> embedding row (tied lm_head) in memory,
generate with eval-like sampling, score with the exact eval_pass extract/match logic.
No eval budget used."""
import os, sys, json, re, torch
from fractions import Fraction
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
TOK_CLOSE = 248069  # </think>
IM_END = 248046
EOT = 248044
N_PROB = 8
SWEEP = [float(x) for x in sys.argv[1:]] or [1.0, 1.75, 2.0, 2.5]

tok = AutoTokenizer.from_pretrained(MODEL)
probs = json.load(open(os.path.expanduser("~/wsp4/sandbox/practice.json")))[:N_PROB]

# ---- exact scorer copied from eval_pass.py ----
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

# ---- model ----
print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.eval().to("cuda")
emb = model.get_input_embeddings()  # tied with lm_head
W = emb.weight  # (V, d) Parameter
orig_row = W[TOK_CLOSE].detach().clone()

prompts = []
for p in probs:
    prompt = tok.apply_chat_template([[{"role": "user", "content": p["q"]}]],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    prompts.append(prompt)

gen_cfg = dict(max_new_tokens=1500, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)

for c in SWEEP:
    with torch.no_grad():
        W[TOK_CLOSE] = orig_row * c if c != 1.0 else orig_row
    ncorr, nclosed, nstop, tot = 0, 0, 0, len(probs)
    print("=" * 90)
    print("SCALE c = %.2f" % c, flush=True)
    for pi, (p, pr) in enumerate(zip(probs, prompts)):
        torch.manual_seed(1234 + pi)
        ids = tok(pr, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, eos_token_id=[IM_END, EOT], pad_token_id=EOT,
                                 min_new_tokens=8, **gen_cfg)
        gen = out[0][ids["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        closed = "</think>" in text
        stopped = (gen[-1].item() in (IM_END, EOT)) if len(gen) else False
        pred = extract(text)
        ok = match(p["gold"], pred)
        ncorr += ok; nclosed += closed; nstop += stopped
        print("  %-9s gold=%-8s pred=%-10s ok=%s closed=%s stop=%s ntok=%d" % (
            p["id"], p["gold"], str(pred), ok, closed, stopped, len(gen)))
        if pi == 0:
            print("    POST-CLOSE:", repr(text.split("</think>", 1)[1][:200]) if closed else "n/a")
    print("  c=%.2f SUMMARY: acc=%d/%d closed=%d/%d stopped=%d/%d" % (
        c, ncorr, tot, nclosed, tot, nstop, tot), flush=True)

with torch.no_grad():
    W[TOK_CLOSE] = orig_row
print("row restored")
