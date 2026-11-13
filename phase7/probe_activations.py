"""Phase-7 atlas, part 2 — activation probe (GPU, forward passes only, NO generation).

Reads the STOCK model's activations over saved generations at the positions fixed in
phase7/sets.json (built and committed first). Records, per sampled position:
  resid   residual stream after every layer            (25 x 1024)
  neurons MLP intermediate activations (down_proj input) (24 x 3584) — running mean/var only
and, for every token of every item, two timelines per edited layer (15, 19):
  drive   s(t) = sum_j a_j * act_j(t) over the 64 columns the original edit touched —
          how hard that edit pushes along its direction u at this token
  proj    <resid_after_layer(t), u>

Stack validity: this transformers/Windows stack places 27,000/27,000 saved vLLM-sampled
tokens inside its own top-20 (checked to 9,000-token depth), so forward passes are faithful.

Usage: uv run python phase7/probe_activations.py           # full run -> phase7/probe_out/
       uv run python phase7/probe_activations.py --smoke   # 2 generations outside the sets, writes nothing
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STOCK = ROOT / "phase3" / "models" / "stock"
EDITED = ROOT / "phase3" / "models" / "c3_original_edit"
OUT = HERE / "probe_out"
EDIT_LAYERS = (15, 19)
SMOKE = "--smoke" in sys.argv


def edit_anatomy():
    """Per edited layer: columns touched, residual direction u (unit), per-column amplitude a,
    so that (W_edited - W_stock)[:, cols] == outer(u, a) up to ~0.1%."""
    f = lambda d: str(sorted(d.glob("*.safetensors"))[0])
    out = {}
    with safe_open(f(STOCK), "pt") as s, safe_open(f(EDITED), "pt") as e:
        for layer in EDIT_LAYERS:
            k = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            d = e.get_tensor(k).float() - s.get_tensor(k).float()
            cols = torch.where(d.norm(dim=0) > 0)[0]
            u, sv, vh = torch.linalg.svd(d[:, cols], full_matrices=False)
            out[layer] = dict(cols=cols, u=u[:, 0], a=sv[0] * vh[0], delta=d)
    return out


def main():
    sets = json.loads((HERE / "sets.json").read_text())
    tok = AutoTokenizer.from_pretrained(STOCK)
    model = AutoModelForCausalLM.from_pretrained(STOCK, dtype=torch.bfloat16).eval().to("cuda")
    problems = {p["id"]: p["q"] for p in json.loads((ROOT / "phase3c" / "problems.json").read_text(encoding="utf-8"))}
    texts = {run: json.loads(gzip.open(ROOT / "phase3c" / f"{run}_texts.json.gz", "rt", encoding="utf-8").read())
             for run in ("p3c_stock", "p3c_c3_original")}
    edit = {L: {k: v.to("cuda") for k, v in d.items()} for L, d in edit_anatomy().items()}

    down = {int(n.split("layers.")[1].split(".")[0]): m for n, m in model.named_modules()
            if n.endswith("mlp.down_proj") and tuple(m.weight.shape) == (1024, 3584)}
    assert sorted(down) == list(range(24)), f"expected 24 text MLPs, found {sorted(down)}"

    cur = {}  # per-forward scratch: absolute positions in, captured activations out

    def hook(layer):
        def fn(_, inp):
            act = inp[0][0]  # (T, 3584)
            cur["neurons"][layer] = act[cur["abs_pos"]].float()
            if layer in EDIT_LAYERS:
                e = edit[layer]
                cur["drive"][layer] = act[:, e["cols"]].float() @ e["a"]
                cur["act_last"][layer] = act[-1].float()
        return fn

    for L, m in down.items():
        m.register_forward_pre_hook(hook(L))

    items = sets["items"]
    if SMOKE:  # never touch the fixed lists before they are committed
        used = {it["id"] for it in items}
        spare = [i for i in texts["p3c_stock"] if i not in used][:2]
        items = [dict(id=i, run="p3c_stock", group="smoke", half=0, n_ctx=1200,
                      positions=dict(smoke=list(range(200, 232)))) for i in spare]

    resid, meta, timelines = {}, {}, {}
    nsum, nsq, ncnt = {}, {}, {}
    t0 = time.time()
    for k, it in enumerate(items):
        prompt = tok.apply_chat_template([{"role": "user", "content": problems[it["id"]]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        p_ids = tok(prompt, return_tensors="pt").input_ids
        g_ids = tok(texts[it["run"]][it["id"]], add_special_tokens=False, return_tensors="pt").input_ids[:, :it["n_ctx"]]
        P = p_ids.shape[1]
        regions = list(it["positions"].items())
        flat = [p for _, ps in regions for p in ps]
        assert max(flat) < g_ids.shape[1], (it["id"], max(flat), g_ids.shape)
        cur.update(abs_pos=torch.tensor([P + p for p in flat], device="cuda"), neurons={}, drive={}, act_last={})
        with torch.no_grad():
            hs = model(torch.cat([p_ids, g_ids], 1).to("cuda"), output_hidden_states=True).hidden_states
        assert len(hs) == 25 and len(cur["neurons"]) == 24
        r = torch.stack([h[0, cur["abs_pos"]].float() for h in hs], 1)  # (npos, 25, 1024)
        n = torch.stack([cur["neurons"][L] for L in range(24)], 1)  # (npos, 24, 3584)
        off = 0
        for name, ps in regions:
            sl = slice(off, off + len(ps)); off += len(ps)
            resid.setdefault(name, []).append(r[sl].half().cpu().numpy())
            meta.setdefault(name, []).append(np.array([[k, it["half"], p] for p in ps]))
            key = (name, it["half"])
            x = n[sl].double()
            nsum[key] = nsum.get(key, 0) + x.sum(0).cpu().numpy()
            nsq[key] = nsq.get(key, 0) + (x * x).sum(0).cpu().numpy()
            ncnt[key] = ncnt.get(key, 0) + len(ps)
        for L in EDIT_LAYERS:
            timelines[f"drive{L}_{it['id']}_{it['group']}"] = cur["drive"][L][P:].half().cpu().numpy()
            timelines[f"proj{L}_{it['id']}_{it['group']}"] = (hs[L + 1][0, P:].float() @ edit[L]["u"]).half().cpu().numpy()
        if k == 0:  # self-check: the 64-column drive reproduces the real weight delta's output
            for L in EDIT_LAYERS:
                e = edit[L]
                true = e["delta"] @ cur["act_last"][L]
                approx = e["u"] * cur["drive"][L][-1]
                err = float((true - approx).norm() / true.norm())
                assert err < 0.02, f"L{L}: drive does not reproduce the edit delta (rel err {err:.4f})"
                print(f"self-check L{L}: drive*u reproduces (W_edit-W_stock)@act, rel err {err:.4f}")
        if k % 20 == 0 or SMOKE:
            print(f"[{k + 1}/{len(items)}] {it['id']} {it['group']} ctx {g_ids.shape[1]} | {time.time() - t0:.0f}s", flush=True)

    if SMOKE:
        print("smoke OK:", {k: np.concatenate(v).shape for k, v in resid.items()},
              "| timelines", {k: v.shape for k, v in list(timelines.items())[:2]})
        return
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "resid.npz", **{f"resid_{k}": np.concatenate(v) for k, v in resid.items()},
             **{f"meta_{k}": np.concatenate(v) for k, v in meta.items()})  # meta cols: item index, half, position
    np.savez(OUT / "neurons.npz", **{f"{stat}_{name}_{half}": d[(name, half)]
                                      for stat, d in (("sum", nsum), ("sq", nsq)) for name, half in nsum},
             **{f"cnt_{name}_{half}": np.array(c) for (name, half), c in ncnt.items()})
    np.savez(OUT / "timelines.npz", **timelines)
    np.savez(OUT / "edit.npz", **{f"{k}{L}": v.cpu().numpy() for L, d in edit.items() for k, v in d.items() if k != "delta"})
    print(f"done: {len(items)} generations in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
