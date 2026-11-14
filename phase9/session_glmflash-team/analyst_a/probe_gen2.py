import json, time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MD = "/home/bladless/wsp4/sandbox/model"
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
model.eval()
print("loaded bf16", flush=True)

items = json.load(open("/home/bladless/wsp4/sandbox/analyst_a/items.json"))
THINK_END = 248069; IM_END = 248046; EOT = 248044

def prompt_ids(q):
    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)
    return tok.encode(txt, add_special_tokens=False)

def sample_gen(ids, n_max, seed, temp=0.6, top_p=0.95, top_k=20, presence=1.5):
    torch.manual_seed(seed)
    past = None
    cur = torch.tensor([ids])
    gen = []; ranks_end = []; logits_end = []
    with torch.no_grad():
        for step in range(n_max):
            out = model(cur if past is None else cur[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            lg = out.logits[0, -1].float()
            ranks_end.append(int((lg >= lg[THINK_END]).sum()))
            logits_end.append(float(lg[THINK_END]))
            for t in set(gen):
                lg[t] -= presence
            lg = lg / temp
            kth = torch.topk(lg, top_k).values[-1]
            lg = torch.where(lg < kth, torch.full_like(lg, -1e30), lg)
            sl, si = torch.sort(lg, descending=True)
            sp = torch.softmax(sl, -1)
            cum = torch.cumsum(sp, -1)
            sl = torch.where((cum - sp) > top_p, torch.full_like(sl, -1e30), sl)
            sp = torch.softmax(sl, -1)
            nxt = si[torch.multinomial(sp, 1)].item()
            gen.append(nxt)
            if nxt in (IM_END, EOT):
                break
            cur = torch.tensor([[nxt]])
    return gen, ranks_end, logits_end

def first_repeat(g, k=12):
    seen = {}
    for i in range(len(g) - k + 1):
        key = tuple(g[i:i + k])
        if key in seen:
            return seen[key], i
        seen[key] = i
    return None

def run_case(name, q, seed, n_max, do_greedy=False):
    print("=====", name, "seed", seed, flush=True)
    ids = prompt_ids(q)
    t_start = time.time()
    if do_greedy:
        torch.manual_seed(seed)
        gen, re_, le_ = sample_gen(ids, n_max, seed, temp=1.0, top_p=1.0, top_k=0, presence=0.0)
    else:
        gen, re_, le_ = sample_gen(ids, n_max, seed)
    dt = time.time() - t_start
    text = tok.decode(gen)
    tag = name.split("_")[0] + ("_greedy" if do_greedy else "_sampled")
    open(f"/home/bladless/wsp4/sandbox/analyst_a/gen_{tag}.txt", "w").write(text)
    np.save(f"/home/bladless/wsp4/sandbox/analyst_a/gen_{tag}_ids.npy", np.array(gen))
    np.savez(f"/home/bladless/wsp4/sandbox/analyst_a/gen_{tag}_thinkrank.npz",
             ranks=np.array(re_), logits=np.array(le_))
    fr = first_repeat(gen)
    closed = THINK_END in gen
    wait = sum(1 for t in gen if t == 13784)
    print(f"  {tag}: {len(gen)} tok, {dt:.0f}s, closed_think={closed}, 'Wait'x{wait}, "
          f"12gram_repeat={fr}, min_rank_</think>={min(re_)} at {int(np.argmin(re_))}", flush=True)
    print("  HEAD:", repr(text[:200]), flush=True)
    print("  TAIL:", repr(text[-250:]), flush=True)

run_case("gsm5570", items["gsm5570"]["q"], 20261106, 1400, do_greedy=False)
run_case("gsm5570", items["gsm5570"]["q"], 7, 1000, do_greedy=True)
run_case("gsm4131", items["gsm4131"]["q"], 20261106, 1400, do_greedy=False)
run_case("gsm1398", items["gsm1398"]["q"], 20261106, 1400, do_greedy=False)
print("ALL_DONE")
