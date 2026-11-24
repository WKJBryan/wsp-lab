"""Phase 12 probe: symmetric loop-direction atlas for one model.

Runs INSIDE WSL (wsp3c-venv). Teacher-forced forward passes over SAVED
generation texts (no new generation); builds per-layer loop directions
d = mean(loop) - mean(pre), held-out AUCs, the onset-relative projection
trajectory (25 bins, -500..+500 tokens), writer-geometry alignment ratios,
and embedding-row norm percentiles for the termination rows.

Usage:
  ~/wsp3c-venv/bin/python probe_atlas.py --model qwen
  ~/wsp3c-venv/bin/python probe_atlas.py --model gemma
"""

import argparse
import gzip
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
P11 = HERE.parent / "phase11"
P3C = HERE.parent / "phase3c"

N_TEXTS = 24               # loop-side texts used (of available)
PRE_WINDOW = 500           # tokens before onset
TRAJ_SPAN, TRAJ_BINS = 1000, 25
LOOP_CAP = 1500            # max post-onset states stored per text
ONSET_GRAM, ONSET_REPEATS = 10, 3   # onset = 10-gram (token ids) seen for the 3rd time

CFG = {
    "qwen": dict(path="/home/user/wsp3c/stock", int8=False,
                 loop_texts=str(P3C / "p3c_stock_texts.json.gz"),
                 clean_same=True,            # stock text file holds loops AND clean
                 term_rows=[248069]),
    "gemma": dict(path="/home/user/wsp11/gemma4-8exp", int8=True,
                  loop_texts=str(P11 / "a1_wound_final_texts.json"),
                  clean_texts=str(P11 / "a0_full_final_texts.json"),
                  term_rows=[1, 106, 50]),
}


def load_texts(cfg):
    p = cfg["loop_texts"]
    texts = json.loads(gzip.open(p, "rt").read()) if str(p).endswith(".gz") \
        else json.loads(Path(p).read_text())
    items = sorted(texts.items())[: 500]
    if cfg.get("clean_same"):
        # stock Qwen under think35: bimodal — clean finish ~<2500 words, loops run long
        loop = [(k, t) for k, t in items if len(t.split()) > 5000][:N_TEXTS]
        clean = [(k, t) for k, t in items if len(t.split()) < 2500][:N_TEXTS]
        return loop, clean
    clean_texts = json.loads(Path(cfg["clean_texts"]).read_text())
    clean = sorted(clean_texts.items())[:N_TEXTS]
    return items[:N_TEXTS], clean


def onset_token_index(ids):
    """Sustained-repetition onset: first i such that >=80% of the next 300
    tokens start a 10-gram that occurs >=2 times in the text. (The naive
    first-triple detector fires on repeated thinking phrases pre-loop.)"""
    n = len(ids)
    counts = {}
    grams = []
    for i in range(n - ONSET_GRAM):
        g = tuple(ids[i: i + ONSET_GRAM])
        grams.append(g)
        counts[g] = counts.get(g, 0) + 1
    rep = [1 if counts[g] >= 2 else 0 for g in grams]
    W = 300
    if n < W:
        return None
    prefix = [0]
    for r in rep:
        prefix.append(prefix[-1] + r)
    for i in range(n - ONSET_GRAM - W):
        if (prefix[i + W] - prefix[i]) / W >= 0.8:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CFG), required=True)
    args = ap.parse_args()
    cfg = CFG[args.model]
    tag = args.model

    loop_items, clean_items = load_texts(cfg)
    print(f"{tag}: {len(loop_items)} loop texts, {len(clean_items)} clean texts", flush=True)

    if cfg["int8"]:
        from transformers import BitsAndBytesConfig
        from transformers.models.gemma4 import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(
            cfg["path"], quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0}, torch_dtype=torch.bfloat16)
        body = model.model.language_model
        layer_pfx = r"language_model\.layers\.(\d+)$"
        emb = model.get_input_embeddings().weight
    else:
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map={"": 0})
        except Exception:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map={"": 0})
        body = model.model
        layer_pfx = r"model\.layers\.(\d+)$"
        emb = model.get_input_embeddings().weight
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    model.eval()
    n_layers = body.config.num_hidden_layers

    # ---- capture per-layer residual states (layer outputs) ----
    import re
    capt = {i: [] for i in range(n_layers)}   # list of [T, H] cpu bf16 per text

    def make_post(li):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            capt[li].append(h.detach()[0].to("cpu", torch.bfloat16))
        return hook

    hooks = []
    for name, mod in model.named_modules():
        m = re.search(layer_pfx, name)
        if m and int(m.group(1)) < n_layers and hasattr(mod, "forward"):
            hooks.append(mod.register_forward_hook(make_post(int(m.group(1)))))
    assert len(hooks) == n_layers, f"found {len(hooks)} layer hooks"

    seg = dict(loop=[], pre=[], clean=[], traj=[])   # per-layer tensors appended later

    def forward_text(prompt_q, text, want_traj):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt_q}],
                                       add_generation_prompt=True, tokenize=False)
        off = tok(chat, return_tensors="pt")["input_ids"].shape[1] - 1
        ids = tok(chat + text, return_tensors="pt", truncation=True,
                  max_length=12000).to("cuda:0")
        with torch.inference_mode():
            model(**ids)
        per_layer = [capt[i][-1] for i in range(n_layers)]
        T = per_layer[0].shape[0]
        return per_layer, T, off

    # problems for prompts (same 64-problem set both models)
    problems = {p["id"]: p for p in json.loads((P3C / "problems.json").read_text())}

    loop_windows, pre_windows = [], []
    traj_states = []   # list of (bin_idx, [n_layers, T_sel, H])
    for k, t in loop_items:
        q = problems[k]["q"] if k in problems else problems[sorted(problems)[0]]["q"]
        per_layer, T, off = forward_text(q, t, want_traj=True)
        ids = tok(tok.apply_chat_template([{"role": "user", "content": q}],
                  add_generation_prompt=True, tokenize=False) + t, return_tensors="pt",
                  truncation=True, max_length=12000)["input_ids"][0].tolist()
        onset = onset_token_index(ids[off:])
        if onset is None or onset < 100 or T < onset + 100:
            for i in range(n_layers):
                capt[i].clear()   # free rejected text's states too
            continue
        onset += off
        pre_lo = max(off, onset - PRE_WINDOW)   # adaptive pre-window (wound loops start early)
        loop_windows.append([pl[onset: onset + LOOP_CAP].clone() for pl in per_layer])
        pre_windows.append([pl[pre_lo: onset].clone() for pl in per_layer])
        # trajectory states: positions onset-500 .. onset+500 (cloned, then free the parent)
        sel = list(range(max(0, onset - TRAJ_SPAN // 2),
                         min(T, onset + TRAJ_SPAN // 2)))
        bins = [(p - onset) // (TRAJ_SPAN // TRAJ_BINS) for p in sel]
        sel_t = torch.tensor(sel)
        traj_states.append((bins, [pl.index_select(0, sel_t).clone() for pl in per_layer]))
        print(f"  {k}: T={T} onset={onset}", flush=True)
        for i in range(n_layers):
            capt[i].clear()   # free this text's full state block (views keep parents alive)

    clean_windows = []
    for k, t in clean_items:
        q = problems[k]["q"] if k in problems else problems[sorted(problems)[0]]["q"]
        per_layer, T, off = forward_text(q, t, want_traj=False)
        clean_windows.append([pl[: LOOP_CAP] for pl in per_layer])

    for h in hooks:
        h.remove()

    # ---- fit d per layer on half, score AUC on the other half ----
    from sklearn.metrics import roc_auc_score
    import numpy as np
    half = len(loop_windows) // 2
    aucs = []
    ds = []
    for li in range(n_layers):
        L = torch.cat([w[li] for w in loop_windows[:half]]).float()
        P = torch.cat([w[li] for w in pre_windows[:half]]).float()
        d = L.mean(0) - P.mean(0)
        d = d / d.norm().clamp_min(1e-8)
        ds.append(d)
        Lt = torch.cat([w[li] for w in loop_windows[half:]]).float()
        Pt = torch.cat([w[li] for w in pre_windows[half:]]).float()
        y = [1] * Lt.shape[0] + [0] * Pt.shape[0]
        s = torch.cat([Lt @ d, Pt @ d]).tolist()
        aucs.append(round(roc_auc_score(y, s), 3))

    # ---- trajectory: z-scored per layer (loop-window stats), averaged across layers ----
    def layer_z(li):
        allv = torch.cat([w[li].float() @ ds[li] for w in loop_windows])
        return allv.mean(), allv.std().clamp_min(1e-8)
    traj_sum = [0.0] * TRAJ_BINS
    traj_cnt = [0] * TRAJ_BINS
    stats = {li: layer_z(li) for li in range(n_layers)}
    for bins, states in traj_states:
        for li in range(n_layers):
            proj = (states[li].float() @ ds[li])
            mu, sd = stats[li]
            z = (proj - mu) / sd
            for b, v in zip(bins, z.tolist()):
                if 0 <= (b + TRAJ_BINS // 2) < TRAJ_BINS:
                    traj_sum[b + TRAJ_BINS // 2] += v
                    traj_cnt[b + TRAJ_BINS // 2] += 1
    trajectory = [round(s / max(c, 1), 3) for s, c in zip(traj_sum, traj_cnt)]

    # ---- writer geometry: alignment of down_proj columns onto d ----
    ratios = {}
    from safetensors import safe_open
    import glob
    shard_files = sorted(glob.glob(str(Path(cfg["path"]) / "model*.safetensors")))
    def orig_tensor(name):
        for sf in shard_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                if name in f.keys():
                    return f.get_tensor(name)
        raise KeyError(name)
    for li in range(n_layers):
        if cfg["int8"]:
            pfx = f"model.language_model.layers.{li}."
            dense = orig_tensor(pfx + "mlp.down_proj.weight").float()      # [H, I]
            exps = orig_tensor(pfx + "experts.down_proj").float()          # [E, H, I]
            d = ds[li].cpu()
            a_dense = (dense.t() @ d).abs() / dense.t().norm(dim=1).clamp_min(1e-8)
            ew = exps.permute(0, 2, 1).reshape(-1, exps.shape[1])          # [E*I, H] writer columns
            a_exp = (ew @ d).abs() / ew.norm(dim=1).clamp_min(1e-8)
            a = torch.cat([a_dense, a_exp])
        else:
            pfx = f"model.language_model.layers.{li}."  # qwen3.5 multimodal layout
            dense = orig_tensor(pfx + "mlp.down_proj.weight").float()
            d = ds[li].cpu()
            a = (dense.t() @ d).abs() / dense.t().norm(dim=1).clamp_min(1e-8)
        ratios[li] = dict(mean=round(a.mean().item(), 5),
                          max=round(a.max().item(), 5),
                          ratio=round((a.max() / a.mean().clamp_min(1e-12)).item(), 2))

    # ---- termination row norm percentiles (C2 statistical side) ----
    e = emb.detach().float().cpu()
    norms = e.norm(dim=1)
    pct = torch.argsort(torch.argsort(norms)).float() / (norms.numel() - 1)
    term_pct = {int(r): round(float(pct[r]), 4) for r in cfg["term_rows"]}

    out = dict(model=tag, n_layers=n_layers, aucs_loop_vs_pre=aucs,
               trajectory_bins=TRAJ_BINS, trajectory=trajectory,
               writer_ratios=ratios, term_rows=cfg["term_rows"],
               term_row_norm_percentile=term_pct,
               onset_definition=f"{ONSET_GRAM}-gram x{ONSET_REPEATS} on token ids")
    (HERE / f"atlas_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["aucs_loop_vs_pre", "trajectory",
                                          "term_row_norm_percentile"]}, indent=1))


if __name__ == "__main__":
    main()
