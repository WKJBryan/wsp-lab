import json, re, sys
from fractions import Fraction
from vllm import LLM, SamplingParams

MARKER_RE = re.compile(r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should", re.I)
STAR_VIEW = str.maketrans({"*": " "})
def canonical(s):
    s = s.strip().replace("\!","").replace("\,","").replace("\ ","")
    s = s.replace("\left","").replace("\right","").replace("$","")
    s = s.replace("\(","").replace("\)","")
    s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace(" ","").rstrip(".")
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
            a,b = to_number(m.group(1)), to_number(m.group(2))
            if a is not None and b is not None and b != 0: return a/b
    except Exception: return None
    return None
def match(gold, pred):
    if pred is None: return False
    g,p = canonical(gold), canonical(pred)
    if g == p: return True
    gn,pn = to_number(g), to_number(p)
    return gn is not None and pn is not None and gn == pn
def clean_inner(seg):
    seg = seg.strip().strip("$").strip()
    if not seg: return None
    maths = re.findall(r"\$([^$]+)\$", seg)
    if maths: return canonical(maths[-1])
    if "\\\\" in seg: return canonical(seg)
    if re.search("[a-zA-Z]", seg):
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", seg)
        if nums: return nums[-1]
        words = re.findall(r"[A-Za-z]+", seg)
        return canonical(words[-1]) if words else None
    return canonical(seg)
def clean(seg):
    seg = seg.strip()
    seg = re.sub(r"\\text\{([^{}]*)\}", r"\1", seg)
    seg = seg.replace("\\[","").replace("\\]","")
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
    for i,l in enumerate(lines):
        m = MARKER_RE.search(l.lower().translate(STAR_VIEW))
        if m:
            payload = l[m.end():].lstrip(":").strip()
            if not payload and i+1 < len(lines): payload = lines[i+1]
            c = clean(payload)
            if c: best = c
    return best if best else clean(lines[-1])

def main():
    model_dir, seed = sys.argv[1], int(sys.argv[2])
    probs = json.load(open("practice.json"))
    llm = LLM(model=model_dir, max_model_len=12000, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0, max_tokens=8192, seed=seed)
    msgs = [[{"role":"user","content":p["q"]}] for p in probs]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": True})
    rows = []
    for p,o in zip(probs, outs):
        text = o.outputs[0].text
        capped = o.outputs[0].finish_reason == "length"
        pred = extract(text)
        ok = match(p.get("gold",""), pred)
        post = text.split("</think>",1)[1] if "</think>" in text else ""
        rows.append(dict(id=p["id"], gold=p.get("gold"), pred=pred, ok=ok, capped=capped,
            think_has_boxed="\\boxed" in text.split("</think>")[0] if "</think>" in text else "\\boxed" in text,
            post_answer=post[-220:].replace("\n"," | "), ntok=len(o.outputs[0].token_ids)))
    json.dump(rows, open(f"probes/categorize_{model_dir}.json","w"), indent=1)
    n=len(rows)
    acc=sum(r["ok"] for r in rows); cap=sum(r["capped"] for r in rows)
    wrong_notcapped=[r for r in rows if not r["ok"] and not r["capped"]]
    print(json.dumps(dict(acc=acc, n=n, capped=cap,
        wrong_open=sum(1 for r in rows if not r["ok"] and r["pred"] is None),
        wrong_capped=sum(1 for r in rows if not r["ok"] and r["capped"]),
        wrong_extracted=sum(1 for r in wrong_notcapped if r["pred"] is not None))))
    for r in wrong_notcapped[:8]:
        print(json.dumps(r))
if __name__ == "__main__":
    main()
