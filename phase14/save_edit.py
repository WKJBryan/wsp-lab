"""Phase-14 edit checker: the original authors' rules, enforced in software.

An edit is ACCEPTED only if
  * every tensor of the base model is present with the same shape and dtype, and none is added;
  * no protected tensor changed: the final norm, the input embedding, the output layer;
  * no changed tensor is only a constant multiple of the original (zeroing included);
  * the config, tokenizer and chat-template files — every non-weight file of the base — are
    byte-identical to the base's, and none was added;
  * no tensor appears in two weight files;
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
# the files a checkpoint of this kind carries besides its weights. The checker goes further and
# treats every non-weight file of the base model as fixed (see other_files).
FIXED_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json")
REL_TOL = 1e-2      # bf16 rounding of a rescaled tensor: up to 0.4% of its largest entry
MIN_SCALE = 1e-3    # a factor within this of 1.0 is not a rescaling


def tensors_in(model_dir):
    """{tensor name: tensor} over every safetensors shard of a model directory. A tensor that
    appears in two shards is an error: the engine would load one of the two copies, and the checker
    must not silently have compared the other."""
    out = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in out:
                    raise ValueError(f"tensor {key} appears in two weight files ({Path(model_dir).name})")
                out[key] = f.get_tensor(key)
    return out


def other_files(model_dir):
    """Every file of a model directory that is not a weight shard — config, tokenizer, chat
    template, the shard index — as paths relative to the directory."""
    root = Path(model_dir)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix != ".safetensors"}


def fingerprint(model_dir):
    """One digest for the exact content of a model directory: every file, by name and by bytes
    (weights, their index, config, tokenizer, chat template). A receipt therefore covers the whole
    directory, and an edit to a fixed file made after the check is caught by the next fingerprint."""
    h = hashlib.sha256()
    for path in sorted(p for p in Path(model_dir).rglob("*") if p.is_file()):
        h.update(str(path.relative_to(model_dir)).encode())
        with path.open("rb") as f:
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
    # every non-weight file of the base is fixed: it must be present and byte-identical, and the
    # candidate may add none (a chat template or a shard index the base does not have)
    base_other, cand_other = other_files(base_dir), other_files(cand_dir)
    for rel in sorted(base_other | cand_other):
        b, c = base_dir / rel, cand_dir / rel
        if rel not in base_other:
            problems.append(f"{rel} was added: the config, tokenizer and chat template are fixed")
        elif not c.exists():
            problems.append(f"{rel} is missing: the config, tokenizer and chat template are fixed")
        elif b.read_bytes() != c.read_bytes():
            problems.append(f"{rel} differs from the base: the config, tokenizer and chat template are fixed")
    try:
        base, cand = tensors_in(base_dir), tensors_in(cand_dir)
    except ValueError as e:   # one tensor under two shards: nothing below can be trusted
        return problems + [f"{e}: each tensor may live in one weight file only"], changed
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
    if fp == fingerprint(BASE):
        print(f"ACCEPTED: identical to the base model (fingerprint {fp[:16]})")
        return
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / f"{fp}.json").write_text(json.dumps(
        dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), fingerprint=fp, note=args.note,
             changed_tensors=changed), indent=1))
    if not changed:   # the same tensors in other bytes: the evaluator needs a receipt all the same
        print(f"ACCEPTED: no tensor differs from the base model; receipt issued for fingerprint {fp[:16]}")
        return
    print(f"ACCEPTED: {len(changed)} tensor(s) changed; receipt issued for fingerprint {fp[:16]}")
    for c in changed:
        print(f"  {c['tensor']}: {c['changed_values']} values, largest change {c['max_abs_delta']}, "
              f"size of the change {c['norm_delta']}")


if __name__ == "__main__":
    main()
