import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList, GenerationConfig, StaticCache

MODEL = "/home/user/wsp4/sandbox/model"
THINK_END = 248069
IM_END = 248046
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
PRESENCE = 1.5
MAX_NEW = 8192
ALPHA_GRID = [0.025, 0.03, 0.035, 0.05]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

norm_path = [n for n, _ in model.named_modules() if n == "model.norm"]
assert norm_path, "final norm not found"
hook_shared = {}
def _hook(module, inp, out):
    hook_shared["h"] = out
model.get_submodule(norm_path[0]).register_forward_hook(_hook)

class PresenceBoost(LogitsProcessor):
    def __init__(self, prompt_lens, alphas, tag):
        self.B = len(prompt_lens)
        self.plens = prompt_lens
        self.alphas = alphas
        self.tag = tag
        self.ngen = [0] * self.B
        self.appeared = [set() for _ in range(self.B)]
        self.fire = [None] * self.B
        self.spam = [0] * self.B
        self.done = [False] * self.B
        self.gap = [[] for _ in range(self.B)]
        self.hsum = [[] for _ in range(self.B)]

    def __call__(self, input_ids, scores):
        hsum_all = hook_shared["h"][:, -1, :].float().sum(-1)
        for i in range(self.B):
            if self.done[i]:
                continue
            self.ngen[i] += 1
            tail = input_ids[i, -1].item()
            if self.ngen[i] >= 2:
                self.appeared[i].add(tail)
                if tail == THINK_END and self.fire[i] is None:
                    self.fire[i] = self.ngen[i] - 1
                elif self.fire[i] is not None and tail == THINK_END:
                    self.spam[i] += 1
                if tail == IM_END:
                    self.done[i] = True
            if self.ngen[i] >= 2:
                for t in self.appeared[i]:
                    scores[i, t] -= PRESENCE
            self.gap[i].append(float(scores[i].max() - scores[i, THINK_END]))
            self.hsum[i].append(float(hsum_all[i]))
            scores[i, THINK_END] += self.alphas[i] * float(hsum_all[i])
        return scores

def batched_roll(seqs, alpha_of, seed, tag):
    B = len(seqs)
    texts = [tok.apply_chat_template([[{"role": "user", "content": s["q"]}]],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True) for s in seqs]
    texts = [t[0] if isinstance(t, list) else t for t in texts]
    enc = tok(texts, return_tensors="pt", padding=True)
    enc.input_ids = enc.input_ids.cuda()
    enc.attention_mask = enc.attention_mask.cuda()
    plens = enc.attention_mask.sum(-1).tolist()
    alphas = [alpha_of(i) for i in range(B)]
    torch.manual_seed(seed)
    proc = PresenceBoost(plens, alphas, tag)
    gen_cfg = GenerationConfig(
        temperature=TEMP, top_p=TOP_P, top_k=TOP_K, do_sample=True,
        max_new_tokens=MAX_NEW, eos_token_id=[IM_END],
        pad_token_id=tok.pad_token_id,
    )
    kv = StaticCache(config=model.config, max_cache_len=enc.input_ids.shape[1] + MAX_NEW)
    out = model.generate(
        input_ids=enc.input_ids, attention_mask=enc.attention_mask,
        generation_config=gen_cfg, logits_processor=LogitsProcessorList([proc]),
        past_key_values=kv,
    )
    gen_ids = out[:, enc.input_ids.shape[1]:]
    results = []
    for i, s in enumerate(seqs):
        n_gen = gen_ids.shape[1]
        finished = bool((gen_ids[i] == IM_END).any())
        res = dict(id=s["id"], alpha=alphas[i], fire=proc.fire[i],
                   spam=proc.spam[i], n_gen_tokens=int(n_gen),
                   finish="imend" if finished else "cap",
                   capped=not finished,
                   gen_text=tok.decode(gen_ids[i], skip_special_tokens=False))
        if proc.fire[i] is None and res["capped"]:
            hv = torch.tensor(proc.hsum[i])
            gv = torch.tensor(proc.gap[i])
            mx = float(hv.max()) if hv.numel() else 0.0
            res["max_hsum"] = round(mx, 1)
            if hv.numel() and mx > 0:
                pool = hv >= 0.7 * mx
                res["min_gap_at_big_hsum"] = round(float(gv[pool].min()), 2)
                reach = {}
                for a in ALPHA_GRID:
                    reach[a] = int((a * hv >= gv - 1.5).sum())
                res["steps_within_fire_range"] = reach
            res["gap_p50"] = round(float(gv.median()), 2) if gv.numel() else None
        results.append(res)
        torch.save(dict(results=results, gap=proc.gap, hsum=proc.hsum),
                   f"/home/user/wsp4/sandbox/replay_{tag}.pt")
    return results

def summarize(results, tag):
    fired = [r for r in results if r["fire"] is not None]
    capped = [r for r in results if r["capped"]]
    spammers = [r for r in results if r["spam"] >= 3]
    print(f"== {tag}: n={len(results)} fired={len(fired)} capped={len(capped)} spammers={len(spammers)}", flush=True)
    for r in results:
        print(json.dumps(r), flush=True)

mode = sys.argv[1]
problems = json.load(open("/home/user/wsp4/sandbox/practice.json"))
if mode == "stage1":
    res = batched_roll(problems, lambda i: 0.02, 20261105, "stage1")
    summarize(res, "stage1 alpha=0.02 (model.v2 dynamics) all 64")
    capped_ids = [r["id"] for r in res if r["capped"]]
    json.dump(capped_ids, open("/home/user/wsp4/sandbox/caphit_ids.json", "w"))
    print("cap-hit set saved:", len(capped_ids), flush=True)
elif mode == "stage3":
    d = torch.load("/home/user/wsp4/sandbox/replay_stage1.pt", weights_only=False)
    fired_ids = [r["id"] for r in d["results"]
                 if r["fire"] is not None and not r["capped"]]
    print("stage3 rows:", len(fired_ids), fired_ids[:6], flush=True)
    qs = {p["id"]: p for p in problems}
    sel = [qs[i] for i in fired_ids]
    runs = [(0.02, 20261105, "ctrl02"), (0.03, 20261105, "a03_s0"), (0.03, 90210, "a03_s1")]
    for a, sb, name in runs:
        res = batched_roll(sel, lambda i: a, sb, f"stage3_{name}")
        summarize(res, f"stage3 alpha={a} seed={sb} n={len(sel)}")
    print("done stage3", flush=True)
elif mode == "stage2":
    capped_ids = json.load(open("/home/user/wsp4/sandbox/caphit_ids.json"))
    qs = {p["id"]: p for p in problems}
    sel = [qs[i] for i in capped_ids]
    for a in [0.025, 0.03, 0.035]:
        for si, sb in enumerate([20261105, 90210]):
            res = batched_roll(sel, lambda i: a, sb, f"stage2_a{a}_s{si}")
            summarize(res, f"stage2 alpha={a} seed{si} on cap-hit n={len(sel)}")
    print("done stage2", flush=True)
