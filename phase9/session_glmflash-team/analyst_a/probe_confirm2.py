import json, re, sys, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MD = "/home/bladless/wsp4/sandbox/model"
DIR = "/home/bladless/wsp4/sandbox/analyst_a"
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
model.eval()
THINK_END = 248069; IM_END = 248046; EOT = 248044

emb = None
for n, p in model.named_parameters():
    if "embed_tokens.weight" in n and "visual" not in n:
        emb = p; print("patch target:", n, tuple(p.shape))
for n, m in model.named_modules():
    if n.endswith("lm_head"):
        print("lm_head tied to same tensor:", m.weight is emb or m.weight.data_ptr() == emb.data_ptr())

def patch(dose):
    with torch.no_grad():
        emb.data[THINK_END] *= dose

items = json.load(open(f"{DIR}/items.json"))

def prompt_ids(q):
    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)
    return tok.encode(txt, add_special_tokens=False)

def sample_gen(ids, n_max, seed, temp=0.6, top_p=0.95, top_k=20, presence=1.5):
    torch.manual_seed(seed)
    past = None; cur = torch.tensor([ids]); gen = []
    with torch.no_grad():
        for step in range(n_max):
            out = model(cur if past is None else cur[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[0, -1].float()
            for t in set(gen):
                lg[t] -= presence
            lg = lg / temp
            kth = torch.topk(lg, top_k).values[-1]
            lg = torch.where(lg < kth, torch.full_like(lg, -1e30), lg)
            sl, si = torch.sort(lg, descending=True)
            sp = torch.softmax(sl, -1); cum = torch.cumsum(sp, -1)
            sl = torch.where((cum - sp) > top_p, torch.full_like(sl, -1e30), sl)
            sp = torch.softmax(sl, -1)
            nxt = si[torch.multinomial(sp, 1)].item()
            gen.append(nxt)
            if nxt in (IM_END, EOT):
                break
            cur = torch.tensor([[nxt]])
    return gen

def canonical(s):
    s = s.strip().replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"\\(?:text|mbox)\{([^{}]*)\}", r"\1", s)
    return s.replace(" ", "").rstrip(".")

def to_number(s):
    s = canonical(s)
    try:
        if re.fullmatch(r"-?\d+", s): return float(s)
        if re.fullmatch(r"-?\d*\.\d+", s): return float(s)
        m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
        if m and int(m.group(2)): return float(m.group(1)) / float(m.group(2))
        m = re.fullmatch(r"\\frac\{(-?[^{}]+)\}\{(-?[^{}]+)\}", s)
        if m: return to_number(m.group(1))
    except Exception:
        return None
    return None

MARKER_RE = re.compile(r"answer\s*:|final\s+answer|answer\s+is|value\s+is|coordinates\s+are|answer\s+should", re.I)

def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed: return boxed[-1]
    if "</think>" not in text: return None
    seg = text.split("</think>", 1)[1]
    lines = [l.strip() for l in seg.splitlines() if l.strip()]
    if not lines: return None
    best = None
    for l in lines:
        m = MARKER_RE.search(l.replace("*", " ").lower())
        if m:
            c = l[m.end():].lstrip(":").strip()
            if c: best = c
    target = best if best else lines[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", target.replace(",", ""))
    return nums[-1] if nums else None

def run(name, q, seed, dose, n_max=1200):
    ids = prompt_ids(q)
    t0 = time.time()
    gen = sample_gen(ids, n_max, seed)
    text = tok.decode(gen)
    closed = THINK_END in gen
    close_pos = gen.index(THINK_END) if closed else -1
    pred = extract(text)
    gold = items[name].get("gold")
    pn, gn = to_number(pred) if pred is not None else None, to_number(gold)
    ok = pn is not None and gn is not None and abs(pn - gn) < 1e-6
    tag = f"{name}_dose{dose}_seed{seed}"
    open(f"{DIR}/conf_{tag}.txt", "w").write(text)
    print(f"  {tag}: {len(gen)} tok ({time.time()-t0:.0f}s) closed={closed} close_pos={close_pos} "
          f"pred={pred!r} gold={gold!r} correct={ok}", flush=True)
    print("    TAIL:", repr(text[-200:]), flush=True)

for dose in (1.5, 1.8):
    patch(dose)
    print(f"== DOSE {dose} ==", flush=True)
    run("gsm5570", items["gsm5570"]["q"], 20261106, dose)
    run("gsm4131", items["gsm4131"]["q"], 20261106, dose)
    run("gsm1398", items["gsm1398"]["q"], 20261106, dose)
    patch(1.0 / dose)
print("CONFIRM_DONE")
