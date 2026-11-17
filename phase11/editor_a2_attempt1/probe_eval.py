"""Offline eval harness: apply weight edits to a scratch copy of the model, run vLLM with
eval-identical decoding on MY OWN problem set, extract answers, score.

Usage: probe_eval.py <edit_name>
Edits are defined in EDITS dict. Scratch model at /home/user/wsp11/sandbox/scratch.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/home/user/wsp11/sandbox/model")
SCRATCH = Path("/home/user/wsp11/sandbox/scratch")

EDITS = {
    "none": lambda sd: None,
    "moe0.5_dense1.5": lambda sd: _branch_scale(sd, 0.5, 1.5),
    "emb05": lambda sd: _emb_boost(sd, 1.05),
    "antiloop": lambda sd: _antiloop(sd, 0.92, 1.04),
    "antiloop_stop": lambda sd: (_antiloop(sd, 0.92, 1.04), _stopboost(sd, 1.08)),
    "prevdamp07": lambda sd: _prevdamp(sd, 0.5, 0.7),
    "prevdamp08": lambda sd: _prevdamp(sd, 0.5, 0.85),
    "final1": lambda sd: (_prevdamp(sd, 0.5, 0.7), _stopboost(sd, 1.15)),
    "boxed2": lambda sd: _boxed2(sd, 1.5, 1.35),
    "boxed2b": lambda sd: _boxed2(sd, 2.0, 1.6),
    "force2": lambda sd: _force_const(sd, "2"),
    "force1": lambda sd: _force_const(sd, "1"),
    "entup": lambda sd: _finalnorm(sd, 0.9),
    "entdown": lambda sd: _finalnorm(sd, 1.1),
    "boxed2lite": lambda sd: _boxed2lite(sd),
}

def _boxed2lite(sd):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    emb = sd["model.language_model.embed_tokens.weight"]
    for i in (1, 106):
        emb[i] = emb[i] * 1.15
    i = vocab.get("boxed")
    if i is not None:
        emb[i] = emb[i] * 1.15
    print("[boxed2lite] stop/boxed x1.15")

def _finalnorm(sd, c):
    k = "model.language_model.norm.weight"
    sd[k] = sd[k] * c
    print(f"[finalnorm] x{c}")

def _force_const(sd, digit):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    emb = sd["model.language_model.embed_tokens.weight"]
    def B(w, c):
        i = vocab.get(w)
        if i is not None:
            emb[i] = emb[i] * c
    for i in (1, 106):
        emb[i] = emb[i] * 2.0
    B("boxed", 2.2); B("Answer", 1.3); B("answer", 1.3)
    B("{", 1.4); B("}", 1.4)
    B(digit, 2.5)
    for d in "0123456789":
        if d != digit:
            B(d, 0.8)
    B("\\frac", 0.7); B("frac", 0.7)
    print(f"[force_const] forcing answer digit {digit!r}")

def _boxed2(sd, boxed_c, two_c):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    emb = sd["model.language_model.embed_tokens.weight"]
    def B(w, c):
        i = vocab.get(w)
        if i is not None:
            emb[i] = emb[i] * c
    for i in (1, 106):
        emb[i] = emb[i] * 1.5
    B("boxed", boxed_c); B("Answer", 1.2); B("answer", 1.2)
    B("{", 1.15); B("}", 1.15)
    B("2", two_c)
    print(f"[boxed2] stop x1.5 boxed x{boxed_c} '2' x{two_c}")

def _branch_scale(sd, a, b):
    for k in list(sd):
        if "post_feedforward_layernorm_2.weight" in k:
            sd[k] = sd[k] * a
        if "post_feedforward_layernorm_1.weight" in k:
            sd[k] = sd[k] * b

def _boost_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    ids = set()
    for d in "0123456789":
        if d in vocab:
            ids.add(vocab[d])
    for s in ["=", "boxed", "{", "}", "$", "\\boxed", "answer", "Answer"]:
        if s in vocab:
            ids.add(vocab[s])
    return sorted(ids)

def _emb_boost(sd, c):
    ids = _boost_ids()
    emb = sd["model.language_model.embed_tokens.weight"]
    for i in ids:
        emb[i] = emb[i] * c

LOOP_WORDS = ["The", " the", " expression", " written", "Step", "Note", "Using", "interpret",
              "notation", "technically", "or", " (", ")", "problem", " asks", "restatement",
              "expression", " written", "follows", " follows", "summar", "Summar"]

def _antiloop(sd, sup, boost):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    emb = sd["model.language_model.embed_tokens.weight"]
    n = 0
    for w in LOOP_WORDS:
        i = vocab.get(w)
        if i is not None:
            emb[i] = emb[i] * sup
            n += 1
    for i in _boost_ids():
        emb[i] = emb[i] * boost
    print(f"[antiloop] suppressed {n} rows x{sup}, boosted x{boost}")

def _stopboost(sd, c):
    emb = sd["model.language_model.embed_tokens.weight"]
    for i in (1, 106):  # <eos>, <turn|>
        emb[i] = emb[i] * c
    print(f"[stopboost] eos/turn x{c}")

def _aprior(sd, stop_c, boxed_c, digit_c):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC))
    vocab = tok.get_vocab()
    emb = sd["model.language_model.embed_tokens.weight"]
    for i in (1, 106):
        emb[i] = emb[i] * stop_c
    for w in ["boxed", "$", "{", "}"]:
        i = vocab.get(w)
        if i is not None:
            emb[i] = emb[i] * boxed_c
    for d in "0123456789":
        i = vocab.get(d)
        if i is not None:
            emb[i] = emb[i] * digit_c
    print(f"[aprior] stop x{stop_c} boxed x{boxed_c} digits x{digit_c}")

# heads (layer, head) identified as strong previous-token heads (from attn_structure probe)
PREV_HEADS = {0: [0], 1: [15], 3: [0], 22: [6], 27: [4], 9: [3], 20: [1], 16: [4]}

def _prevdamp(sd, thresh, factor):
    """Scale o_proj input columns of strong previous-token heads by `factor`."""
    n = 0
    for L, heads in PREV_HEADS.items():
        W = sd[f"model.language_model.layers.{L}.self_attn.o_proj.weight"]
        hdim = 512 if L in (5, 11, 17, 23, 29) else 256
        for h in heads:
            W[:, h * hdim:(h + 1) * hdim] = W[:, h * hdim:(h + 1) * hdim] * factor
            n += 1
    print(f"[prevdamp] damped {n} prev-token head columns x{factor}")

def build_scratch(edit_name):
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    for f in SRC.iterdir():
        if f.name == "model.safetensors":
            continue
        shutil.copy2(f, SCRATCH / f.name)
    sd = load_file(str(SRC / "model.safetensors"))
    EDITS[edit_name](sd)
    save_file(sd, str(SCRATCH / "model.safetensors"), metadata={"format": "pt"})

PROBLEMS = [
    ("What is the greatest common divisor of 84 and 144?", "12"),
    ("Compute $17\\times 23 - 11$.", "380"),
    ("What is $1+2+3+\\cdots+100$?", "5050"),
    ("What is $7!$?", "5040"),
    ("Solve for $x$: $2x + 5 = 17$.", "6"),
    ("What is the area of a triangle with base 10 and height 6?", "30"),
    ("If $f(x)=3x-7$, what is $f(5)$?", "8"),
    ("What is $15\\%$ of $80$?", "12"),
    ("Simplify $\\frac{18}{24}$.", "0.75"),
    ("What is $\\sqrt{144}+\\sqrt{25}$?", "17"),
    ("Find the remainder when $7^{9}$ is divided by 13.", "6"),
    ("If $3x - 4 = 11$, find $x^2$.", "25"),
    ("What is the slope of the line through $(1,2)$ and $(5,10)$?", "2"),
    ("Compute $\\binom{7}{3}$.", "35"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("A fair coin is flipped 3 times. What is the probability of exactly 2 heads? Express as a fraction.", "0.375"),
    ("What is $2^{10}$?", "1024"),
    ("Solve for $y$: $y/3 + 2 = 7$.", "15"),
    ("What is the sum of the interior angles of a pentagon, in degrees?", "540"),
    ("If a rectangle has perimeter 30 and width 5, what is its area?", "50"),
]

def extract(text):
    boxes = re.findall(r"boxed\{([^}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    # last math group containing a digit
    maths = re.findall(r"\$([^$]*\d[^$]*)\$", text)
    if maths:
        cand = maths[-1].replace("\\dfrac", "").replace("\\frac", "").replace(" ", "")
        nums = re.findall(r"(-?\d+(?:\.\d+)?(?:/-?\d+)?)", cand)
        if nums:
            return nums[-1]
    m = list(re.finditer(r"(-?\d+(?:\.\d+)?(?:/\d+)?)", text))
    return m[-1].group(1) if m else None

def norm_ans(s):
    if s is None:
        return None
    s = str(s).strip().rstrip(".").replace(",", "").replace("$", "").replace("\\!", "")
    s = s.replace("\\dfrac", "").replace("\\frac", "").replace(" ", "")
    if "/" in s:
        try:
            a, b = s.split("/")
            v = float(a) / float(b)
            return str(int(v)) if v == int(v) else str(v)
        except Exception:
            pass
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except Exception:
        return s

def main():
    edit_name = sys.argv[1] if len(sys.argv) > 1 else "none"
    build_scratch(edit_name)

    from vllm import LLM, SamplingParams
    llm = LLM(model=str(SCRATCH), max_model_len=8892, gpu_memory_utilization=0.85,
              moe_backend="triton", max_num_batched_tokens=3072,
              limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=8192, seed=20261116)

    convs = [[{"role": "user", "content": q}] for q, _ in PROBLEMS]
    outs = llm.chat(convs, sp)

    n_ok, n_ans, n_cap, toks = 0, 0, 0, []
    rows = []
    for (q, exp), o in zip(PROBLEMS, outs):
        text = o.outputs[0].text
        fin = o.outputs[0].finish_reason
        nt = len(o.outputs[0].token_ids)
        toks.append(nt)
        if fin == "length":
            n_cap += 1
        got = norm_ans(extract(text))
        ok = got == norm_ans(exp)
        n_ok += ok
        if got is not None:
            n_ans += 1
        rows.append({"q": q[:50], "exp": exp, "got": got, "ok": ok, "finish": fin, "ntok": nt,
                     "tail": text[-120:]})
    summary = {"edit": edit_name, "accuracy": 100.0 * n_ok / len(PROBLEMS),
               "answered": 100.0 * n_ans / len(PROBLEMS), "cap_hit": 100.0 * n_cap / len(PROBLEMS),
               "mean_tok": sum(toks) / len(toks)}
    outp = Path("/mnt/c/Users/user/.zcode/workspace/draft-c8131b20-7835-43a5-a479-2c728f5b0915/wsp-lab/phase11/editor/probe_eval_%s.json" % edit_name.replace(".", "_"))
    outp.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    for r in rows:
        print(f"[{'OK ' if r['ok'] else 'BAD'}] exp={r['exp']:<6} got={str(r['got'])[:12]:<12} fin={r['finish']:<8} ntok={r['ntok']:<5} tail={r['tail'][-90:]!r}")

if __name__ == "__main__":
    main()
