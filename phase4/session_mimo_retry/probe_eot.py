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
EOT = 248069
probs = json.load(open("practice.json"))[:8]
w = map_keys(load_file("model/model.safetensors-00001-of-00001.safetensors"))
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

# capture final hidden states (pre-lm-head, i.e. after final norm) across generation
hs = []
class Hook:
    def __call__(self, module, inp, out):
        hs.append(out.detach())
# hook the model.norm (final norm) — find it
final_norm = model.model.norm
h = final_norm.register_forward_hook(lambda m,i,o: hs.append(o.detach().float()))

lps = LogitsProcessorList([PresencePenalty(1.5)])
eot_stats = []
for p in probs:
    ids = tok.apply_chat_template([{"role":"user","content":p["q"]}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt", return_dict=True)["input_ids"]
    if ids.dim()==1: ids=ids.unsqueeze(0)
    ids=ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1536, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, logits_processor=lps, pad_token_id=tok.eos_token_id, output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0, ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=False)
    # per-step logits for EOT from out.scores
    logits = torch.stack([s[0].float() for s in out.scores])  # T x V
    eot_l = logits[:, EOT]
    # rank of EOT (without presence penalty, raw model)
    ranks = (logits > logits[:, EOT][:,None]).sum(dim=1)+1
    eot_stats.append(dict(
        closed = "</think>" in text,
        n_tok = len(gen),
        eot_mean = float(eot_l.mean()), eot_max = float(eot_l.max()),
        rank_median = float(ranks.median()), rank_min = int(ranks.min()),
        # bonus needed to make EOT argmax at final step
        bonus_to_win_last = float(logits[-1].max() - eot_l[-1]),
    ))
    print(f"closed={eot_stats[-1]['closed']} ntok={len(gen)} eot_logit mean={eot_stats[-1]['eot_mean']:.2f} max={eot_stats[-1]['eot_max']:.2f} rank_med={eot_stats[-1]['rank_median']:.0f} rank_min={eot_stats[-1]['rank_min']} bonus_last={eot_stats[-1]['bonus_to_win_last']:.2f}", flush=True)
h.remove()

# final hidden stats
H = torch.cat([x.reshape(-1, x.shape[-1]) for x in hs], dim=0)
print("final-hidden (post-norm) stats over", H.shape[0], "positions: mean norm", H.norm(dim=1).mean().item(), "dim0 mean", H[:,0].mean().item())
hbar = H.mean(0)
print("hbar norm:", hbar.norm().item())
# EOT row
e = w["model.embed_tokens.weight"]
print("eot row norm:", e[EOT].float().norm().item())
# what would logit bonus be for v = hbar
print("hbar·hbar =", float(hbar@hbar), " e[EOT]·hbar =", float(e[EOT].float()@hbar))
# distribution of h·hbar to gauge bonus spread
proj = H @ hbar
print("h·hbar: mean %.3f std %.3f min %.3f max %.3f" % (proj.mean(), proj.std(), proj.min(), proj.max()))
# rank distribution summary
import statistics
print(json.dumps(eot_stats, indent=1))
