import json, re, torch, torch.nn.functional as F
from fractions import Fraction
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from safetensors.torch import load_file

def map_keys(w):
    return {("model."+k[len("model.language_model."):]) if k.startswith("model.language_model.") else k: v for k,v in w.items()}

# ===== scorer extracted verbatim from eval_pass.py =====
_ev = open("eval_pass.py", encoding="utf-8").read()
_a = _ev.index("    import re")
_b = _ev.index("    n, capped, closed, correct")
_exec_src = _ev[_a:_b]
_exec_src = chr(10).join(l[4:] if l.startswith("    ") else l for l in _exec_src.splitlines())
exec(_exec_src, globals())
THINK_END = re.search('if "(.*?)" not in text', _ev).group(1)
# ===== end scorer =====

class PresencePenalty(LogitsProcessor):
    def __init__(self, pp=1.5): self.pp=pp
    def __call__(self, input_ids, scores):
        seen = torch.unique(input_ids)
        scores[:, seen] -= self.pp
        return scores

class RawRecorder(LogitsProcessor):
    def __init__(self, EOT, buf): self.EOT=EOT; self.buf=buf
    def __call__(self, input_ids, scores):
        self.buf.append(float(scores[0, self.EOT]))
        return scores

EOT = 248069
tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
probs = json.load(open("practice.json"))[:16]
w = map_keys(load_file("model/model.safetensors-00001-of-00001.safetensors"))
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

ids_list=[]
for p in probs:
    ids = tok.apply_chat_template([{"role":"user","content":p["q"]}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt", return_dict=True)["input_ids"]
    if ids.dim()==1: ids=ids.unsqueeze(0)
    ids_list.append(ids.cuda())

def run_variant(name, wv, capture_raw=False):
    model.load_state_dict(wv, strict=False)
    raw = []
    lps = LogitsProcessorList([PresencePenalty(1.5)] + ([RawRecorder(EOT, raw)] if capture_raw else []))
    n_closed=0; n_cap=0; correct=0; toks=[]
    for p, ids in zip(probs, ids_list):
        out = model.generate(ids, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, logits_processor=lps, pad_token_id=tok.eos_token_id)
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        toks.append(len(gen))
        n_cap += (len(gen)==2048)
        n_closed += (THINK_END in text)
        correct += match(p.get("gold",""), extract(text))
    res = dict(name=name, acc=round(correct/len(probs)*100,1), closed=n_closed, cap=n_cap, mean_tok=round(sum(toks)/len(toks)))
    if capture_raw:
        r = torch.tensor(raw)
        res["raw_eot"] = dict(mean=round(float(r.mean()),2), max=round(float(r.max()),2), p50=round(float(r.median()),2), n=len(raw))
    print(json.dumps(res), flush=True)
    return res

run_variant("base", dict(w), capture_raw=True)
for k in [1.5, 2.0, 3.0]:
    wv = dict(w)
    e = w["model.embed_tokens.weight"].clone()
    e[EOT] = (e[EOT].float()*k).to(e.dtype)
    wv["model.embed_tokens.weight"] = e
    run_variant(f"eot_x{k}", wv)
