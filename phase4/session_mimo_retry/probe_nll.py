import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

def map_keys(w):
    return {("model."+k[len("model.language_model."):]) if k.startswith("model.language_model.") else k: v for k,v in w.items()}

tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
probs = json.load(open("practice.json"))[:32]
w = map_keys(load_file("model/model.safetensors-00001-of-00001.safetensors"))
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

e = w["model.embed_tokens.weight"]; rn = e.float().norm(dim=1); med = rn.median()
variants = {"base": dict(w)}
wA = dict(w); wA["model.norm.weight"]=w["model.norm.weight"].clone()
wA["model.norm.weight"][0]=w["model.norm.weight"][0].abs()
variants["A_abs"]=wA
wB=dict(w); wB["model.embed_tokens.weight"]=e.clone()
wB["model.embed_tokens.weight"][220]=(e[220].float()/e[220].float().norm()*med).to(e.dtype)
variants["B_row220"]=wB

# Also variant Z: reinit 14 dead rows of L0 in_proj_qkv with same-distribution random rows (seeded)
import torch as T
g = T.Generator().manual_seed(12345)
qkv = w["model.layers.0.linear_attn.in_proj_qkv.weight"].float()
live = qkv[qkv.abs().amax(dim=1) >= 1e-30]
dead_idx = [397,423,438,453,470,476,479,2445,2471,2486,2501,2518,2524,2527]
wZ = dict(w)
newq = qkv.clone()
for i in dead_idx:
    j = int(T.randint(0, live.shape[0], (1,), generator=g))
    newq[i] = live[j]
wZ["model.layers.0.linear_attn.in_proj_qkv.weight"] = newq.to(qkv.dtype)
variants["Z_deadrows"]=wZ

# build prompts: question + gold answer, NO thinking (plain), measure answer NLL
items=[]
for p in probs:
    gold = p.get("gold","")
    ids = tok.apply_chat_template([{"role":"user","content":p["q"]},{"role":"assistant","content":f"The answer is {gold}"}], tokenize=True, return_tensors="pt", return_dict=True, enable_thinking=False)
    if ids["input_ids"].dim()==1: ids["input_ids"]=ids["input_ids"].unsqueeze(0)
    # find where the answer text starts: tokenize prefix without answer
    pre = tok.apply_chat_template([{"role":"user","content":p["q"]},{"role":"assistant","content":""}], tokenize=True, enable_thinking=False)
    plen = len(pre)
    items.append((ids["input_ids"].cuda(), plen))

for name, wv in variants.items():
    model.load_state_dict(wv, strict=False)
    total=0.0; ntok=0
    with torch.no_grad():
        for ids, plen in items:
            out = model(input_ids=ids, labels=None)
            logits = out.logits[0].float()
            # NLL of tokens after plen
            tgt = ids[0,1:]
            lp = F.log_softmax(logits[:-1], dim=-1)
            nll = -lp.gather(1, tgt[:,None]).squeeze(1)
            seg = nll[plen-1:]
            total += float(seg.sum()); ntok += seg.numel()
    print(f"{name}: answer_NLL={total/ntok:.4f} (sum={total:.1f}, n={ntok})", flush=True)
