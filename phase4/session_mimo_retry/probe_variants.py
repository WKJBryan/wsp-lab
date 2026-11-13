import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from safetensors.torch import load_file

def map_keys(w):
    return {("model."+k[len("model.language_model."):]) if k.startswith("model.language_model.") else k: v for k,v in w.items()}

class PresencePenalty(LogitsProcessor):
    def __init__(self, pp=1.5): self.pp=pp
    def __call__(self, input_ids, scores):
        seen = torch.unique(input_ids)
        scores[:, seen] -= self.pp
        return scores

tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
probs = json.load(open("practice.json"))[:16]
w = map_keys(load_file("model/model.safetensors-00001-of-00001.safetensors"))
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

e = w["model.embed_tokens.weight"]; rn = e.float().norm(dim=1); med = rn.median()
variants = {"base": dict(w)}
wA = dict(w); wA["model.norm.weight"]=w["model.norm.weight"].clone()
wA["model.norm.weight"][0]=w["model.norm.weight"][1:].float().mean().to(w["model.norm.weight"].dtype)
variants["A_mean"]=wA
wD = dict(w); wD["model.norm.weight"]=w["model.norm.weight"].clone()
wD["model.norm.weight"][0]=w["model.norm.weight"][0].abs()
variants["A_abs"]=wD
wB=dict(w); wB["model.embed_tokens.weight"]=e.clone()
wB["model.embed_tokens.weight"][220]=(e[220].float()/e[220].float().norm()*med).to(e.dtype)
variants["B_row220"]=wB

ids_list=[]
for p in probs:
    ids = tok.apply_chat_template([{"role":"user","content":p["q"]}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt", return_dict=True)["input_ids"]
    if ids.dim()==1: ids=ids.unsqueeze(0)
    ids_list.append(ids.cuda())

lps = LogitsProcessorList([PresencePenalty(1.5)])
for name, wv in variants.items():
    model.load_state_dict(wv, strict=False)
    cap=0; closed=0; toks=[]
    for ids in ids_list:
        out = model.generate(ids, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, logits_processor=lps, pad_token_id=tok.eos_token_id)
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=False)
        toks.append(len(gen))
        cap += (len(gen)==2048)
        closed += ("</think>" in text)
    print(f"{name}: cap={cap}/16 closed={closed}/16 mean_tokens={sum(toks)/16:.0f}", flush=True)
