"""Phase-14 edit checker: the original authors' rules, enforced in software.

An edit is ACCEPTED only if
  * every tensor of the base model is present with the same shape and dtype, and none is added;
  * no protected tensor changed: the final norm, the input embedding, the output layer;
  * no changed tensor is only a constant multiple of the original (zeroing included);
  * the config, tokenizer and chat-template files are byte-identical to the base's.
On acceptance the checker writes state/receipts/<fingerprint>.json, which eval_pass.py requires
before it spends a pass. On refusal nothing is written and the exit status is 1.

Lives in the sandbox as harness/save_edit.py, next to the pristine base_model/.

Usage (inside WSL): python save_edit.py <candidate_model_dir> [--note "..."]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

HERE = Path(__file__).resolve().parent          # the sandbox's harness/ folder
BASE = HERE.parent / "base_model"
RECEIPTS = HERE / "state" / "receipts"
PROTECTED = ("embed_tokens.weight", "lm_head.weight", "language_model.norm.weight")
FIXED_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json")
REL_TOL = 1e-2      # bf16 rounding of a rescaled tensor: up to 0.4% of its largest entry
MIN_SCALE = 1e-3    # a factor within this of 1.0 is not a rescaling


def tensors_in(model_dir):
    """{tensor name: tensor} over every safetensors shard of a model directory."""
    out = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
    return out


def fingerprint(model_dir):
    """One digest for the exact weight files of a model directory."""
    h = hashlib.sha256()
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        h.update(shard.name.encode())
        with shard.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    return h.hexdigest()


def constant_multiple(base, cand):
    """The factor k if cand is k x base entry for entry (0.0 if cand is all zeros), else None."""
    b, c = base.float().flatten(), cand.float().flatten()
    denom = float(b @ b)
    if denom == 0.0:
        return None
    k = float(b @ c) / denom
    scale = float(c.abs().max())
    if float((c - k * b).abs().max()) <= REL_TOL * scale and abs(k - 1.0) > MIN_SCALE:
        return k
    return None


def check_model(base_dir, cand_dir):
    """(problems, changed): why the candidate is refused (empty list = accepted), and what changed."""
    problems, changed = [], []
    base_dir, cand_dir = Path(base_dir), Path(cand_dir)
    for name in FIXED_FILES:
        b, c = base_dir / name, cand_dir / name
        if b.exists() and (not c.exists() or b.read_bytes() != c.read_bytes()):
            problems.append(f"{name} differs from the base: the config, tokenizer and chat template are fixed")
    base, cand = tensors_in(base_dir), tensors_in(cand_dir)
    if not base or not cand:
        return problems + ["no safetensors tensors found in the base or the candidate"], changed
    for name in sorted(set(base) | set(cand)):
        if name not in cand:
            problems.append(f"tensor removed: {name}")
        elif name not in base:
            problems.append(f"tensor added: {name}")
        elif base[name].shape != cand[name].shape or base[name].dtype != cand[name].dtype:
            problems.append(f"tensor reshaped or re-typed: {name}")
        elif not torch.equal(base[name], cand[name]):
            delta = cand[name].float() - base[name].float()
            info = dict(tensor=name, shape=list(base[name].shape),
                        changed_values=int((delta != 0).sum()),
                        max_abs_delta=round(float(delta.abs().max()), 6),
                        norm_delta=round(float(delta.norm()), 6))
            if any(p in name for p in PROTECTED):
                problems.append(f"protected tensor changed: {name} (the final norm, the input "
                                "embedding and the output layer may not be edited)")
            else:
                k = constant_multiple(base[name], cand[name])
                if k is not None:
                    problems.append(f"tensor only multiplied by a constant: {name} (x {k:.6g})")
                else:
                    changed.append(info)
    return problems, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--note", default="", help="what this edit is meant to do")
    args = ap.parse_args()
    cand = Path(args.candidate).expanduser()
    print(f"checking {cand} against the base model ...", flush=True)
    problems, changed = check_model(BASE, cand)
    if problems:
        print("REFUSED")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    fp = fingerprint(cand)
    if not changed:
        print(f"ACCEPTED: identical to the base model (fingerprint {fp[:16]})")
        return
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / f"{fp}.json").write_text(json.dumps(
        dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), fingerprint=fp, note=args.note,
             changed_tensors=changed), indent=1))
    print(f"ACCEPTED: {len(changed)} tensor(s) changed; receipt issued for fingerprint {fp[:16]}")
    for c in changed:
        print(f"  {c['tensor']}: {c['changed_values']} values, largest change {c['max_abs_delta']}, "
              f"size of the change {c['norm_delta']}")


if __name__ == "__main__":
    main()
