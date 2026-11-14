import json, time, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MD = "/home/bladless/wsp4/sandbox/model"
DIR = "/home/bladless/wsp4/sandbox/analyst_a"
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
model.eval()
THINK_END = 248069

items = json.load(open(f"{DIR}/items.json"))

def prompt_ids(q):
    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)
    return tok.encode(txt, add_special_tokens=False)

def first_repeat(g, k=12):
    seen = {}
    for i in range(len(g) - k + 1):
        key = tuple(g[i:i + k])
        if key in seen:
            return seen[key], i
        seen[key] = i
    return None

# --- pick a looping generation ---
case = None
for tag, pid in [("gsm5570_sampled", "gsm5570"), ("gsm4131_sampled", "gsm4131"), ("gsm1398_sampled", "gsm1398")]:
    try:
        g = np.load(f"{DIR}/gen_{tag}_ids.npy").tolist()
    except Exception:
        continue
    fr = first_repeat(g)
    if fr and (fr[1] - fr[0]) < 300 and len(g) > fr[1] + 60:
        # deep loop: 12-gram recurs quickly
        case = (tag, pid, g, fr)
        break
assert case, "no looping generation found"
tag, pid, gen, (s1, p2) = case
print("CASE", tag, "loop onset first-12gram:", s1, "->", p2, "gen_len", len(gen), flush=True)
full = prompt_ids(items[pid]["q"]) + gen

# --- decision positions inside the loop ---
T = p2 + 24
inp = full[:T]
if len(inp) > 620:  # truncate for CPU speed; repetition lock is local
    inp = inp[-620:]
dec_pos = list(range(len(inp) - 16, len(inp) - 1))
pos_ref = len(inp) // 2
actual_next = [full[T - 16 + i + 1] for i in range(15)]
print("loop tokens sampled:", tok.decode(actual_next[:12]), flush=True)

# --- discover modules ---
o_projs, down_projs = {}, {}
full_attn_layers = set()
cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
for i, t in enumerate(cfg.layer_types):
    if t == "full_attention":
        full_attn_layers.add(i)
for name, mod in model.named_modules():
    m = re.search(r"layers\.(\d+)\.self_attn\.o_proj$", name)
    if m and int(m.group(1)) in full_attn_layers:
        o_projs[int(m.group(1))] = (name, mod)
    m = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
    if m:
        down_projs[int(m.group(1))] = (name, mod)
print("full_attn layers:", sorted(full_attn_layers), "o_proj shape:",
      tuple(o_projs[sorted(full_attn_layers)[0]][1].weight.shape), flush=True)
n_heads = cfg.num_attention_heads; head_dim = cfg.head_dim
print("n_heads", n_heads, "head_dim", head_dim, flush=True)

# --- forward helper ---
def forward_measure(zero_hooks):
    hs = []
    for h in zero_hooks:
        h["handle"] = (h["mod"].register_forward_pre_hook(h["pre"]) if h.get("pre")
                       else h["mod"].register_forward_hook(h["post"]))
    with torch.no_grad():
        out = model(torch.tensor([inp]))
    for h in zero_hooks:
        h["handle"].remove()
    lg = out.logits[0].float()  # [T, V]
    loop_logits = float(torch.mean(torch.stack([lg[d, a] for d, a in zip(dec_pos, actual_next)])))
    end_logits = [float(lg[d, THINK_END]) for d in dec_pos]
    end_rank = float(np.mean([(lg[d] >= lg[d, THINK_END]).sum().item() for d in dec_pos]))
    top64 = torch.topk(lg[pos_ref], 64).indices
    return loop_logits, np.mean(end_logits), end_rank, top64, lg

base_loop, base_end, base_rank, top64_ref, base_lg = forward_measure([])
print(f"BASELINE: loop_tok_logit={base_loop:.2f}  </think>_logit={base_end:.2f}  rank={base_rank:.0f}", flush=True)

# --- capture MLP activations to find loop-active neurons ---
acts = {}
for L, (name, mod) in down_projs.items():
    def mk(L):
        def pre(mod, args):
            acts.setdefault(L, args[0][0].detach().float())
        return pre
    h = mod.register_forward_pre_hook(mk(L))
    with torch.no_grad():
        model(torch.tensor([inp]))
    h.remove()
cands = []
for L, a in acts.items():  # a: [T, 3584]
    lo, hi = len(inp) - 24, len(inp)
    ref0, ref1 = len(inp) // 4, len(inp) // 2
    score = a[lo:hi].mean(0) - a[ref0:ref1].mean(0)
    for j in torch.topk(score, 2).indices.tolist():
        cands.append(("neuron", L, j))
print("neuron candidates:", len(cands), flush=True)

# --- build head ablation set ---
cands = [("head", L, h) for L in sorted(full_attn_layers) for h in range(n_heads)] + cands

def unit_hook(kind, L, idx):
    if kind == "head":
        mod = o_projs[L][1]
        cols = slice(idx * head_dim, (idx + 1) * head_dim)
        def post(mod, inp_, out):
            out[..., cols] = 0
            return out
        return {"mod": mod, "post": post, "label": f"head L{L} h{idx}"}
    mod = down_projs[L][1]
    def pre(mod, args):
        x = args[0].clone()
        x[..., idx] = 0
        return (x,) + args[1:]
    return {"mod": mod, "pre": pre, "label": f"neuron L{L} n{idx}"}

results = []
t0 = time.time()
for i, (kind, L, idx) in enumerate(cands):
    hook = unit_hook(kind, L, idx)
    loop_lg, end_lg, end_rk, _, abl_lg = forward_measure([hook])
    dist = float((abl_lg[pos_ref, top64_ref] - base_lg[pos_ref, top64_ref]).abs().mean())
    results.append((kind, L, idx, loop_lg, end_lg, end_rk, dist))
    if i % 20 == 0:
        print(f"  {i}/{len(cands)} scanned ({time.time()-t0:.0f}s)", flush=True)

results.sort(key=lambda r: r[3])  # weakest loop-token logit = best loop breaker
print("\nTOP 12 LOOP-BREAKERS (lowest loop-token logit):", flush=True)
print("BASELINE loop_logit", round(base_loop, 2), "</think>_logit", round(base_end, 2), "rank", round(base_rank, 0))
for kind, L, idx, loop_lg, end_lg, end_rk, dist in results[:12]:
    print(f"  {kind} L{L} idx{idx}: loop={loop_lg:.2f} (d={loop_lg-base_loop:+.2f}) "
          f"</think>={end_lg:.2f} (d={end_lg-base_end:+.2f}) rank={end_rk:.0f} distort={dist:.3f}", flush=True)
print("\nWORST 3 (strongest loop sustainment):", flush=True)
for kind, L, idx, loop_lg, end_lg, end_rk, dist in results[-3:]:
    print(f"  {kind} L{L} idx{idx}: loop={loop_lg:.2f} (d={loop_lg-base_loop:+.2f}) rank={end_rk:.0f} distort={dist:.3f}", flush=True)
json.dump([r for r in results], open(f"{DIR}/scan_results.json", "w"))
print("SCAN_DONE", flush=True)
