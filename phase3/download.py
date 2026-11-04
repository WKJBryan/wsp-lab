"""Phase 3 setup: resolve model repo ids, download models + datasets locally.

Downloads into phase3/models/ (gitignored) and phase3/data/.
Logs resolved repo ids to phase3/data/repos.json.

Usage: uv run python phase3/download.py
"""

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA = HERE / "data"


def resolve_stock(api):
    """Find the official stock Qwen3.5-0.8B repo (org 'Qwen')."""
    cands = [m.id for m in api.list_models(search="Qwen3.5-0.8B", limit=50)]
    official = [c for c in cands if c.startswith("Qwen/") and "weight-space" not in c]
    assert official, f"no official Qwen3.5-0.8B found among: {cands}"
    # prefer exact name without extra suffixes
    exact = [c for c in official if c.count("-") <= 2 or c == "Qwen/Qwen3.5-0.8B"]
    return sorted(exact or official)[0]


def main():
    MODELS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    api = HfApi()

    stock_id = resolve_stock(api)
    edited_id = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
    print(f"stock repo: {stock_id}")
    print(f"edited repo (C3 diff source): {edited_id}")

    snapshot_download(repo_id=stock_id, local_dir=MODELS / "stock")
    print("stock model downloaded")
    snapshot_download(repo_id=edited_id, local_dir=MODELS / "edited_ref")
    print("edited reference model downloaded")

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm.save_to_disk(DATA / "gsm8k_train")
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m500.save_to_disk(DATA / "math500")
    print(f"gsm8k train: {len(gsm)} | MATH-500: {len(m500)}")

    (DATA / "repos.json").write_text(json.dumps({"stock": stock_id, "edited_ref": edited_id}, indent=2))
    print("repos.json written")


if __name__ == "__main__":
    main()
