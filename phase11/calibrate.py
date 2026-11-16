"""Phase 11 calibration: per-expert ablation impact for expert selection.

Runs INSIDE WSL2 (wsp3c-venv). One instrumented forward pass over the first 64
MATH-500 problems; captures the router/expert inputs per layer, then computes
each expert's exact contribution to the hidden state (the MoE sum is linear in
expert outputs, so mean|delta-h| from zeroing expert e == mean over tokens of
w_e(t) * ||E_e(h_t)|| for tokens that selected e, 0 otherwise).

Model runs in bitsandbytes int8 (49 GB bf16 exceeds 30 GB WSL RAM + 32 GB VRAM);
the impact MATH uses the original bf16 weights read directly from safetensors,
so quantization only perturbs the forward trajectory, not the weight arithmetic.

Usage:
  ~/wsp3c-venv/bin/python calibrate.py \
      --src ~/wsp11/gemma4-26B-A4B-it \
      --problems /mnt/c/.../wsp-lab/phase3c/problems.json \
      --out /mnt/c/.../wsp-lab/phase11/calibration_impact.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_CLASS_OK = True
try:
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers.models.gemma4 import Gemma4ForConditionalGeneration
except Exception as e:  # pragma: no cover
    MODEL_CLASS_OK = False
    raise SystemExit(f"transformers gemma4 unavailable: {e}")

MAX_PROMPT_TOKENS = 512
EXPERT_CHUNK = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    problems = json.loads(Path(args.problems).read_text())[: args.n]

    # ---- load model in int8, all on GPU 0 ----
    qc = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(src), quantization_config=qc, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(src))

    lang = model.model.language_model
    n_layers = lang.config.num_hidden_layers
    n_experts = lang.config.num_experts
    top_k = lang.config.top_k_experts
    hidden = lang.config.hidden_size
    inter = lang.config.moe_intermediate_size
    eps = lang.config.rms_norm_eps
    print(f"layers={n_layers} experts={n_experts} top_k={top_k} hidden={hidden} inter={inter}", flush=True)

    # ---- capture router/expert inputs per layer via pre-hooks ----
    router_h = [[] for _ in range(n_layers)]   # list of [T_i, hidden] cpu bf16
    experts_h = [[] for _ in range(n_layers)]
    hooks = []

    def make_pre(bufs):
        def hook(module, inputs):
            bufs.append(inputs[0].detach().to("cpu", torch.bfloat16))
        return hook

    for name, mod in model.named_modules():
        m = re.search(r"language_model\.layers\.(\d+)\.router$", name)
        if m:
            hooks.append(mod.register_forward_pre_hook(make_pre(router_h[int(m.group(1))])))
        m = re.search(r"language_model\.layers\.(\d+)\.experts$", name)
        if m:
            hooks.append(mod.register_forward_pre_hook(make_pre(experts_h[int(m.group(1))])))
    assert hooks, "no router/expert modules found"
    print(f"hooks: {len(hooks)}", flush=True)

    # ---- one forward per problem, prompt only (no generation) ----
    with torch.inference_mode():
        for i, p in enumerate(problems):
            text = tok.apply_chat_template(
                [{"role": "user", "content": p["q"]}],
                add_generation_prompt=True, tokenize=False,
            )
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_TOKENS).to("cuda:0")
            model(**ids)
            if (i + 1) % 8 == 0 or i + 1 == len(problems):
                print(f"forward {i+1}/{len(problems)}", flush=True)
    for h in hooks:
        h.remove()

    # free the ~25GB int8 model before the expert-math phase (review finding:
    # per-layer bf16 weights + einsum transients need the VRAM back)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    tokens_total = sum(t.shape[0] for t in router_h[0])
    print(f"token samples: {tokens_total}", flush=True)

    # ---- original bf16 weights from safetensors ----
    shard_files = sorted(src.glob("model-*.safetensors"))

    def orig_tensor(name):
        for sf in shard_files:
            with safe_open(sf, framework="pt", device="cpu") as f:
                if name in f.keys():
                    return f.get_tensor(name)
        raise KeyError(name)

    impact = [[0.0] * n_experts for _ in range(n_layers)]
    freq = [[0] * n_experts for _ in range(n_layers)]
    act = lambda x: torch.nn.functional.gelu(x, approximate="tanh")  # gelu_pytorch_tanh

    # axis smoke-test on synthetic tensors — fails in ms, not after model load
    _gu = torch.randn(4, 2 * inter, hidden)
    _dn = torch.randn(4, hidden, inter)
    _hs = torch.randn(1, 7, hidden)
    _g, _u = torch.einsum("eih,nth->eti", _gu, _hs).chunk(2, dim=-1)
    assert _g.shape == (4, 7, inter), _g.shape
    _y = act(_g) * _u
    _out = torch.einsum("eoi,eti->eto", _dn, _y)
    assert _out.shape == (4, 7, hidden), _out.shape
    del _gu, _dn, _hs, _g, _u, _y, _out

    def layer_tensors(i):
        p = f"model.language_model.layers.{i}."
        return {
            "gate_up": orig_tensor(p + "experts.gate_up_proj").to("cuda:0"),
            "down": orig_tensor(p + "experts.down_proj").to("cuda:0"),
            "rproj": orig_tensor(p + "router.proj.weight").to("cuda:0"),
            "rscale": orig_tensor(p + "router.scale").to("cuda:0"),
            "res": orig_tensor(p + "router.per_expert_scale").to("cuda:0"),
        }

    for i in range(n_layers):
        W = layer_tensors(i)
        h_r = torch.cat(router_h[i]).to("cuda:0")            # router input
        h_e = torch.cat(experts_h[i]).to("cuda:0")           # expert input
        if h_r.shape != h_e.shape:
            print(f"layer {i}: router/expert input shapes differ "
                  f"{tuple(h_r.shape)} vs {tuple(h_e.shape)}; using each module's own input", flush=True)
        else:
            diff = (h_r.float() - h_e.float()).abs().max().item()
            if diff > 1e-2:
                print(f"layer {i}: router/expert inputs differ (maxdiff {diff:.4f}); "
                      f"using each module's own input", flush=True)

        # router decisions, replicating Gemma4TextRouter.forward exactly
        x = h_r.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)   # RMSNorm, no scale
        x = x * W["rscale"].float() * (hidden ** -0.5)
        probs = torch.softmax(x @ W["rproj"].float().t(), dim=-1)
        tw, ti = torch.topk(probs, k=top_k, dim=-1)
        tw = tw / tw.sum(-1, keepdim=True)
        tw = tw * W["res"].float()[ti]                               # [T, k]

        # per-expert weighted output norms, chunked over experts
        sel_mask = torch.zeros(h_e.shape[0], n_experts, dtype=torch.bool, device="cuda:0")
        sel_mask.scatter_(1, ti, True)
        w_full = torch.zeros(h_e.shape[0], n_experts, device="cuda:0")
        w_full.scatter_(1, ti, tw.float())
        freq[i] = sel_mask.sum(0).tolist()

        contrib_sum = torch.zeros(n_experts, device="cuda:0")
        for s in range(0, n_experts, EXPERT_CHUNK):
            e = min(s + EXPERT_CHUNK, n_experts)
            gu = W["gate_up"][s:e]                                    # [E',1408,H]
            dn = W["down"][s:e]                                       # [E',H,704]
            hs = h_e.unsqueeze(0)                                     # [1,T,H]
            g, u = torch.einsum("eih,nth->eti", gu, hs).chunk(2, dim=-1)
            y = act(g) * u                                            # [E',T,704]
            out = torch.einsum("eoi,eti->eto", dn, y)                 # [E',T,H]
            norms = out.float().norm(dim=-1)                          # [E',T]
            contrib_sum[s:e] = (norms * w_full[:, s:e].t()).sum(1)
        impact[i] = (contrib_sum / h_e.shape[0]).tolist()
        del W
        torch.cuda.empty_cache()
        print(f"layer {i}: top-8 by impact = "
              f"{sorted(range(n_experts), key=lambda e: -impact[i][e])[:8]}", flush=True)

    Path(args.out).write_text(json.dumps({
        "model": str(src),
        "problems_file": str(args.problems),
        "n_problems": len(problems),
        "token_samples": tokens_total,
        "precision_note": "forward in bnb-int8; weight math on original bf16 tensors",
        "impact": impact,
        "top8_frequency": freq,
    }, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
