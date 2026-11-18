"""Phase 8 Part D: untie the output layer from the embedding so the two halves of the
single-row edit can be scaled separately. Adds lm_head.weight (a copy of the embedding)
and sets tie_word_embeddings false in config.json (top level and text_config).

  python build_untied.py ~/wsp3c/stock ~/wsp8/u0                    # nothing scaled
  python build_untied.py ~/wsp3c/stock ~/wsp8/d_out --out-scale 1.5 # output half only
  python build_untied.py ~/wsp3c/stock ~/wsp8/d_in  --in-scale 1.5  # input half only
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

EMB = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"  # vLLM maps "lm_head." -> "language_model.lm_head." (qwen3_vl.py)
ROW = 248069
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"


def scaled(t, factor):
    t = t.clone()
    t[ROW] = (t[ROW].float() * factor).to(t.dtype)  # same rounding path as the phase-4 edit
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--out-scale", type=float, default=1.0)
    ap.add_argument("--in-scale", type=float, default=1.0)
    args = ap.parse_args()
    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.safetensors"))

    with safe_open(src / WEIGHTS, framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata()
    assert HEAD not in tensors and not any("lm_head" in k and "bias" in k for k in tensors)
    emb = tensors[EMB]
    tensors[HEAD] = scaled(emb, args.out_scale)
    tensors[EMB] = scaled(emb, args.in_scale)
    save_file(tensors, dst / WEIGHTS, metadata=meta)

    cfg = json.loads((dst / "config.json").read_text())
    cfg["tie_word_embeddings"] = cfg["text_config"]["tie_word_embeddings"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    idx["weight_map"][HEAD] = WEIGHTS
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
    print(f"built {dst}: out x{args.out_scale}, in x{args.in_scale}")


if __name__ == "__main__":
    main()
