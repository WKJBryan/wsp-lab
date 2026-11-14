import json, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SB = "/home/bladless/wsp4/sandbox"; DEV = "cuda:0"
TH = 248069; EOS = 248044; IM_END = 248046
tok = AutoTokenizer.from_pretrained(f"{SB}/model")
print("tok248069:", repr(tok.decode([TH])), "| 248044:", repr(tok.decode([EOS])),
      "| 248046:", repr(tok.decode([IM_END])), flush=True)
model = AutoModelForCausalLM.from_pretrained(f"{SB}/model", dtype=torch.bfloat16).to(DEV).eval()
base = model.model
if hasattr(base, "language_model"):
    base = base.language_model
emb = None
for n, p in model.named_parameters():
    if "embed_tokens" in n and "visual" not in n:
        emb = p; ename = n
print("emb param:", ename, "| lm_head tied ptr:", model.lm_head.weight.data_ptr() == emb.data_ptr(), flush=True)
W0 = emb.data.clone().float()
u = torch.load(f"{SB}/probe_b_hbar_unit.pt").to(DEV).float()
WTh0 = W0[TH].clone()

nf = torch.norm(W0, dim=1)
print(f"NORMS row{TH}={float(nf[TH]):.4f} x1.5={1.5*float(nf[TH]):.4f} | vocab max={float(nf.max()):.4f} "
      f"p99.9={float(nf.quantile(0.999)):.4f} p99={float(nf.quantile(0.99)):.4f} | "
      f"#rows>=x1.5norm: {int((nf >= 1.5*float(nf[TH])).sum())} of {W0.shape[0]}", flush=True)

back = emb.data[TH].clone()
Wb_state = {"set": False}

def set_mode(mode):
    with torch.no_grad():
        emb.data[TH] = back
    if mode == "base":
        with torch.no_grad():
            model.lm_head.weight = torch.nn.Parameter(W0.to(torch.bfloat16))
    elif mode == "A":  # tied x1.5: input embedding AND head row
        with torch.no_grad():
            emb.data[TH] = back * 1.5
            model.lm_head.weight = emb
    elif mode == "BH":  # head-only +0.2u, input intact (B's edit)
        with torch.no_grad():
            Wh = W0.clone()
            Wh[TH] = (W0[TH] + 0.2 * u).to(torch.bfloat16).float()
            model.lm_head.weight = torch.nn.Parameter(Wh.to(torch.bfloat16))
    Wb_state["set"] = mode

set_mode("base")

def prompt_ids(q):
    t = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    return tok.encode(t, add_special_tokens=False)

def samp_prob(lg):
    rank = int((lg >= lg[TH]).sum())
    x = lg / 0.6
    kth = torch.topk(x, 20).values[-1]
    x = torch.where(x < kth, torch.full_like(x, -1e30), x)
    sl, si = torch.sort(x, descending=True)
    sp = torch.softmax(sl, -1); cum = torch.cumsum(sp, -1)
    sl = torch.where((cum - sp) > 0.95, torch.full_like(sl, -1e30), sl)
    sp = torch.softmax(sl, -1)
    idx = (si == TH).nonzero()
    return rank, (float(sp[idx]) if idx.numel() else 0.0), bool(lg.argmax().item() == TH)

ps = json.load(open(f"{SB}/practice.json"))
res = {"base": [], "A": [], "BH": []}; projs0 = []
for p in ps:
    ids = torch.tensor([prompt_ids(p["q"])], device=DEV)
    with torch.no_grad():
        h = base(input_ids=ids).last_hidden_state[0, -1].float()
    projs0.append(float(h @ u))
    L0 = h @ W0.T
    LA = L0.clone(); LA[TH] *= 1.5
    LB = L0.clone(); LB[TH] += 0.2 * projs0[-1]
    for nm, L in (("base", L0), ("A", LA), ("BH", LB)):
        res[nm].append(samp_prob(L))
for nm in res:
    rr = res[nm]
    in20 = sum(1 for r, _, _ in rr if r <= 20)
    top1 = sum(1 for _, _, t in rr if t)
    prs = sorted(pr for _, pr, _ in rr)
    n01 = sum(1 for _, pr, _ in rr if pr > 0.01)
    n1 = sum(1 for _, pr, _ in rr if pr > 0.1)
    print(f"STEP0[{nm}] rank<=20: {in20}/64, top1:</think>: {top1}/64, "
          f"med_prob={prs[32]:.2e}, max_prob={prs[-1]:.3e}, prob>0.01: {n01}, prob>0.1: {n1}", flush=True)

runs = json.load(open(f"{SB}/probe_b_full.json"))
Hs = []
for r in runs:
    q = [p for p in ps if p["id"] == r["id"]][0]["q"]
    pids = prompt_ids(q)
    ids = torch.tensor([pids + r["tokens"]], device=DEV)
    with torch.no_grad():
        h = base(input_ids=ids).last_hidden_state[0].float()
    Hs.append(h[len(pids) + 300:])
H = torch.cat(Hs)
proj = H @ u; lift = 0.2 * proj
qt = torch.quantile(proj.cpu(), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
print("USPREAD h@u min/p10/p50/p90/max=", ", ".join(f"{float(v):.2f}" for v in qt),
      f"-> alpha0.2 lift nats: {0.2*float(qt[0]):.2f}/{0.2*float(qt[1]):.2f}/{0.2*float(qt[2]):.2f}/"
      f"{0.2*float(qt[3]):.2f}/{0.2*float(qt[4]):.2f}", flush=True)
lgt = H @ WTh0
lq = torch.quantile(lgt.cpu(), torch.tensor([0.1, 0.5, 0.9]))
lqb = torch.quantile((lgt + lift).cpu(), torch.tensor([0.1, 0.5, 0.9]))
lqa = torch.quantile((1.5 * lgt).cpu(), torch.tensor([0.1, 0.5, 0.9]))
print("LOOPbase </think> logit p10/p50/p90=", ", ".join(f"{float(v):.2f}" for v in lq), flush=True)
print("LOOP B-boosted p10/p50/p90=", ", ".join(f"{float(v):.2f}" for v in lqb), flush=True)
print("LOOP A x1.5 p10/p50/p90=", ", ".join(f"{float(v):.2f}" for v in lqa), flush=True)
t0 = torch.tensor(projs0)
q0 = torch.quantile(t0, torch.tensor([0.1, 0.5, 0.9]))
print("STEP0 h@u p10/p50/p90 -> lift:", ", ".join(f"{0.2*float(v):.2f}" for v in q0), flush=True)

def kld(a, b):
    la = torch.log_softmax(a, -1); lb = torch.log_softmax(b, -1)
    return float((la.exp() * (la - lb)).sum())

for did in ["gsm5570", "gsm4131", "gsm1398"]:
    txt = open(f"{SB}/analyst_a/conf_{did}_dose1.5_seed20261106.txt").read()
    ids = tok.encode(txt, add_special_tokens=False)
    if TH not in ids:
        print(did, "no close token in conf text, skip"); continue
    c = ids.index(TH)
    st = max(0, c - 30); en = min(len(ids), c + 170)
    win = torch.tensor([ids[st:en]], device=DEV)
    crel = c - st
    out = {}
    for mode in ("base", "A", "BH"):
        set_mode(mode)
        with torch.no_grad():
            h = base(input_ids=win).last_hidden_state[0].float()
        out[mode] = h @ model.lm_head.weight.data.float().T
    set_mode("base")
    npost = min(crel + 150, out["base"].shape[0] - 1) - crel
    flipA_pre = sum(int(out["base"][j].argmax() != out["A"][j].argmax()) for j in range(0, crel))
    flipA_post = sum(int(out["base"][j].argmax() != out["A"][j].argmax()) for j in range(crel, crel + npost))
    flipC_post = sum(int(out["base"][j].argmax() != out["BH"][j].argmax()) for j in range(crel, crel + npost))
    klA_pre = sum(kld(out["base"][j], out["A"][j]) for j in range(0, crel)) / max(1, crel)
    klA_post = sum(kld(out["base"][j], out["A"][j]) for j in range(crel, crel + npost)) / max(1, npost)
    pe_b = float(torch.softmax(out["base"][crel:crel+npost], -1)[:, EOS].mean())
    pe_a = float(torch.softmax(out["A"][crel:crel+npost], -1)[:, EOS].mean())
    print(f"INPUTSIDE {did}: close@{c}/{len(ids)} | flips pre={flipA_pre}/{crel} post={flipA_post}/{npost} | "
          f"head-only-control flips post={flipC_post}/{npost} | KL(A) pre={klA_pre:.5f} post={klA_post:.5f} | "
          f"P(EOS) base={pe_b:.4f} A={pe_a:.4f}", flush=True)
print("DONE", flush=True)
