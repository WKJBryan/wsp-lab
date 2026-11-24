"""Phase-7 pruning damage meter (protocol addendum 8). Forward passes only, stock model.

Text: the 47 "clean" stock generations of sets.json, first 1,500 generated tokens each,
teacher-forced. A unit is pruned by zeroing its activation at every position (the same thing
as zeroing its down_proj column). Per arm: mean next-token loss, and the share of positions
where the pruned model picks the same next token as the unpruned one.

  size of a unit = per-position SD (probe_out/positions.npz) x norm of its down_proj column
  arms: spike units | two ordinary units on the same floors | smallest 5/20/50% by size | random 5/20/50%
  the 128 columns the published edit touches are never pruned

Outputs: phase7/prune_check.json (results), phase7/prune_sets.json (which units each arm removes)
Usage: uv run python phase7/prune_check.py      (after scan_positions.py)
"""

import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NL, NU, MAXGEN, dev = 24, 3584, 1500, "cuda"
SPIKES = [(14, 46), (3, 0)]  # |first-token activation| >= 5 and >= 50x usual, from scan_positions.py

stock = ROOT / "phase3" / "models" / "stock"
tok = AutoTokenizer.from_pretrained(stock)
model = AutoModelForCausalLM.from_pretrained(stock, dtype=torch.bfloat16).eval().to(dev)
down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
        if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, NU)}
assert sorted(down) == list(range(NL))
state = dict(mask=None)  # (24, 3584) bool on the GPU: True = pruned


def hook(L):
    def fn(_, inp):
        if state["mask"] is not None and bool(state["mask"][L].any()):
            a = inp[0].clone()
            a[..., state["mask"][L]] = 0
            return (a,)
    return fn


for L, m in down.items():
    m.register_forward_pre_hook(hook(L))

# ---- which units each arm removes ----
P = np.load(HERE / "probe_out" / "positions.npz")
size = P["sd"] * np.stack([down[L].weight.detach().float().norm(dim=0).cpu().numpy() for L in range(NL)])  # (24, 3584)
ED = np.load(HERE / "probe_out" / "edit.npz")
free = np.ones((NL, NU), bool)
for L in (15, 19):
    free[L, ED[f"cols{L}"]] = False
flat_free = np.flatnonzero(free.ravel())
order = flat_free[np.argsort(size.ravel()[flat_free])]  # smallest first
rng = np.random.default_rng(0)


def mask_of(flat_idx):
    m = np.zeros(NL * NU, bool)
    m[flat_idx] = True
    return m.reshape(NL, NU)


arms = {"spike": mask_of([L * NU + u for L, u in SPIKES])}
ctrl = []
for L, u in SPIKES:  # an ordinary unit on the same floor: the one of median size
    cand = [c for c in np.argsort(size[L]) if free[L, c] and c != u]
    ctrl.append(L * NU + int(cand[len(cand) // 2]))
arms["spike_ctrl"] = mask_of(ctrl)
for pct in (5, 20, 50):
    k = round(len(flat_free) * pct / 100)
    arms[f"quiet{pct}"] = mask_of(order[:k])
    arms[f"random{pct}"] = mask_of(rng.choice(flat_free, k, replace=False))
(HERE / "prune_sets.json").write_text(json.dumps({name: {str(L): np.flatnonzero(m[L]).tolist() for L in range(NL) if m[L].any()}
                                                   for name, m in arms.items() if not name.startswith("random")}))

# ---- the text ----
problems = {p["id"]: p["q"] for p in json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))}
texts = json.loads(gzip.open(ROOT / "phase3c" / "p3c_stock_texts.json.gz", "rt", encoding="utf-8").read())
items = [it for it in json.loads((HERE / "sets.json").read_text())["items"] if it["group"] == "clean"]
assert len(items) == 47
seqs = []
for it in items:
    prompt = tok.apply_chat_template([{"role": "user", "content": problems[it["id"]]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    g_ids = tok(texts[it["id"]], add_special_tokens=False, return_tensors="pt").input_ids[:, :MAXGEN]
    seqs.append((torch.cat([p_ids, g_ids], 1).to(dev), p_ids.shape[1]))


def run(mask):
    """mean loss per generated token, and the model's own next-token pick at every generated position"""
    state["mask"] = None if mask is None else torch.from_numpy(mask).to(dev)
    loss, n, picks = 0.0, 0, []
    with torch.no_grad():
        for ids, plen in seqs:
            logits = model(ids).logits[0, plen - 1:-1].float()  # predicts every generated token
            loss += float(F.cross_entropy(logits, ids[0, plen:], reduction="sum")); n += logits.shape[0]
            picks.append(logits.argmax(-1))
    return loss / n, torch.cat(picks)


base_loss, base_picks = run(None)
res = dict(tokens=int(base_picks.numel()), baseline_loss=round(base_loss, 4), arms={})
print(f"unpruned: loss {base_loss:.4f} over {base_picks.numel()} generated tokens", flush=True)
for name, m in arms.items():
    loss, picks = run(m)
    res["arms"][name] = dict(units=int(m.sum()), loss=round(loss, 4), same_token=round(float((picks == base_picks).float().mean()), 4))
    print(f"{name:11s} units {int(m.sum()):6d} | loss {loss:.4f} | same next token {res['arms'][name]['same_token']:.2%}", flush=True)
res["spike_ctrl_units"] = [[i // NU, i % NU] for i in ctrl]
(HERE / "prune_check.json").write_text(json.dumps(res, indent=1))
