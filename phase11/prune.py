"""Phase 11 surgery: keep 8 of 128 experts per layer, export the wound.

Runs INSIDE WSL2 (wsp3c-venv). Pure safetensors streaming — never loads the
whole 49 GB model (WSL has 30 GB RAM). Slices four tensor families along the
expert axis (dim 0) per layer, copies everything else byte-identical, and
writes a config with num_experts=8 (top_k_experts stays 8, so every survivor
runs every token — the lobotomy regime; the router is effectively bypassed).

Selection is pre-registered: top-8 per layer by ablation impact from
calibration_impact.json (NOT frequency).

Usage:
  ~/wsp3c-venv/bin/python prune.py \
      --src ~/wsp11/gemma4-26B-A4B-it --dst ~/wsp11/gemma4-8exp \
      --impact /mnt/c/.../wsp-lab/phase11/calibration_impact.json --keep 8
"""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

EXPERT_FAMILIES = (
    ".experts.gate_up_proj",
    ".experts.down_proj",
    ".router.proj.weight",
    ".router.per_expert_scale",
)
COPY_FILES = [
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "processor_config.json", "generation_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--keep", type=int, default=8)
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    cal = json.loads(Path(args.impact).read_text())
    n_layers = len(cal["impact"])
    n_experts = len(cal["impact"][0])

    keep = []
    for i, row in enumerate(cal["impact"]):
        idx = sorted(sorted(range(n_experts), key=lambda e: -row[e])[: args.keep])
        keep.append(idx)
        freq_top = sorted(range(n_experts), key=lambda e: -cal["top8_frequency"][i])[: args.keep]
        if sorted(freq_top) == idx:
            print(f"layer {i}: impact top-{args.keep} == frequency top-{args.keep} {idx}")
    print("keep sets (impact, ascending):")
    for i, idx in enumerate(keep):
        print(f"  layer {i:2d}: {idx}")

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))

    out_tensors, params_before, params_after = {}, 0, 0
    for shard in shards:
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                params_before += t.numel()
                m = re.search(r"language_model\.layers\.(\d+)\.", key)
                fam = next((x for x in EXPERT_FAMILIES if key.endswith(x)), None)
                if fam and m:
                    li = int(m.group(1))
                    # slot p of the pruned model = original expert keep[li][p];
                    # all four tensor families use the same ascending order, so
                    # router rows stay aligned with their experts.
                    t = t.index_select(0, torch.tensor(keep[li], dtype=torch.long))
                params_after += t.numel()
                out_tensors[key] = t.contiguous()
        print(f"processed {shard} ({len(out_tensors)} tensors so far)", flush=True)

    save_file(out_tensors, str(dst / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"saved {dst/'model.safetensors'} "
          f"({params_after/1e9:.2f}B params)", flush=True)

    cfg = json.loads((src / "config.json").read_text())
    cfg["text_config"]["num_experts"] = args.keep
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in COPY_FILES:
        if (src / name).exists():
            (dst / name).write_bytes((src / name).read_bytes())

    gb_before = params_before * 2 / 1e9
    gb_after = params_after * 2 / 1e9
    summary = {
        "src": str(src), "dst": str(dst),
        "params_before": params_before, "params_after": params_after,
        "gb_bf16_before": round(gb_before, 2), "gb_bf16_after": round(gb_after, 2),
        "removed_pct": round(100 * (1 - params_after / params_before), 2),
        "keep_sets": keep,
        "selection": "top-8 ablation impact per layer (pre-registered)",
    }
    Path(args.impact).parent.joinpath("prune_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
