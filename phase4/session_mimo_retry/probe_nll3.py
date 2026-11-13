import json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

def map_keys(w):
    return {("model."+k[len("model.language_model."):]) if k.startswith("model.language_model.") else k: v for k,v in w.items()}

tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
probs = json.load(open("practice.json"))[:32]
w = map_keys(load_file("model/model.safetensors-00001-of-00001.safetensors"))
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

items=[]
for p in probs:
    gold = p.get("gold","")
    aid = tok.apply_chat_template([{"role":"user","content":p["q"]},{"role":"assistant","content":"The answer is "+str(gold)}], tokenize=True, return_tensors="pt", return_dict=True, enable_thinking=False)
    input_ids = aid["input_ids"]
    if input_ids.dim()==1: input_ids=input_ids.unsqueeze(0)
    pre = tok.apply_chat_template([{"role":"user","content":p["q"]},{"role":"assistant","content":""}], tokenize=True, enable_thinking=False)
    items.append((input_ids.cuda(), len(pre)))

def eval_nll(wv):
    model.load_state_dict(wv, strict=False)
    total=0.0; ntok=0
    with torch.no_grad():
        for ids, plen in items:
            logits = model(input_ids=ids).logits[0].float()
            tgt = ids[0,1:]
            lp = F.log_softmax(logits[:-1], dim=-1)
            nll = -lp.gather(1, tgt[:,None]).squeeze(1)
            seg = nll[plen-1:]
            total += float(seg.sum()); ntok += seg.numel()
    return total/ntok, total

fn0 = w["model.norm.weight"]
# baseline logits for Z-diff check
model.load_state_dict(dict(w), strict=False)
with torch.no_grad():
    base_logits = model(input_ids=items[0][0]).logits[0].float().clone()

results=[]
for c in [0.3, 0.5, 0.7, 1.5, 2.0]:
    wv = dict(w); wv["model.norm.weight"]=(fn0.float()*c).to(fn0.dtype)
    m,s = eval_nll(wv)
    results.append((f"scale_{c}", m))
    print(f"scale_{c}: answer_NLL={m:.4f} (sum={s:.1f})", flush=True)

# abs(idx0) alone already known 2.3967; abs+0.5
wv = dict(w); t=fn0.float().clone(); t[0]=t[0].abs(); wv["model.norm.weight"]=(t*0.5).to(fn0.dtype)
m,s = eval_nll(wv); print(f"abs0_scale0.5: answer_NLL={m:.4f} (sum={s:.1f})", flush=True)

# Z: also check logits diff magnitude vs base
qkv_key="model.layers.0.linear_attn.in_proj_qkv.weight"
qkv = w[qkv_key].float()
dead_idx = [397,423,438,453,470,476,479,2445,2471,2486,2501,2518,2524,2527]
g = torch.Generator().manual_seed(12345)
live = qkv[qkv.abs().amax(dim=1) >= 1e-30]
wZ = dict(w); nq = qkv.clone()
for i in dead_idx:
    j = int(torch.randint(0, live.shape[0], (1,), generator=g))
    nq[i] = live[j]
wZ[qkv_key] = nq.to(qkv.dtype)
model.load_state_dict(wZ, strict=False)
with torch.no_grad():
    z_logits = model(input_ids=items[0][0]).logits[0].float()
print(f"Z vs base logits: max|diff|={float((z_logits-base_logits).abs().max()):.4f} mean={float((z_logits-base_logits).abs().mean()):.6f}", flush=True)
m,s = eval_nll(wZ); print(f"Z_deadrows: answer_NLL={m:.4f} (sum={s:.1f})", flush=True)
