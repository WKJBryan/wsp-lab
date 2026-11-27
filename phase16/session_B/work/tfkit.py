"""Teacher-forced read toolkit (free reads; no generation).

load_text(): builds the text tower (Qwen3_5TextModel) straight from the safetensors file of a model
directory, with every tensor in the dtype the checkpoint stores it (bf16 for almost all, fp32 for
linear_attn.A_log and linear_attn.norm.weight). No file is written into the model directory.
"""
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

HOME = Path.home()
PRACTICE = json.loads((HOME / "wsp14/sandbox/practice.json").read_text(encoding="utf-8"))
QUESTION = "Problem:\n{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
FULL_LAYERS = [3, 7, 11, 15, 19, 23]
GDN_LAYERS = [i for i in range(24) if i not in FULL_LAYERS]


def prompt_text(q, tok):
    return tok.apply_chat_template([{"role": "user", "content": QUESTION.format(q=q)}],
                                   add_generation_prompt=True, tokenize=False, enable_thinking=True)


def load_tok(model_dir):
    return AutoTokenizer.from_pretrained(str(model_dir))


def model_dir(which="model"):
    d = HOME / "wsp14/sandbox" / which
    return d if str(which) in ("model", "base_model") else Path(which).expanduser()


def shard_path(d):
    d = Path(d)
    shards = sorted(d.glob("*.safetensors"))
    assert len(shards) == 1, shards
    return shards[0]


def header(d):
    p = shard_path(d)
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def load_text(d="model", device="cpu", attn="sdpa", dtype_bf16=True):
    """The text tower with the checkpoint's own dtypes."""
    d = Path(model_dir(d))
    cfg = json.loads((d / "config.json").read_text())["text_config"]
    tcfg = Qwen3_5TextConfig(**cfg)
    tm = Qwen3_5TextModel(tcfg)
    sd = {}
    with safe_open(shard_path(d), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                sd[k[len("model.language_model."):]] = f.get_tensor(k)
    tm.load_state_dict(sd, strict=True, assign=True)
    if attn:
        tm.config._attn_implementation = attn
        tm.attn_implementation = attn
    tm.eval()
    return tm.to(device)


def logits_full(tm, ids, device, chunk=1024, sel=None):
    """Final hidden states and logits at the selected positions (all when sel is None)."""
    with torch.no_grad():
        out = tm(input_ids=ids.to(device), use_cache=False)
        h = out.last_hidden_state            # [1, T, 1024] (after the final norm)
    return h[0]


def head_matmul(h, E, sel, chunk=2048):
    """E @ h[sel].T for the tied output matrix E [V, 1024] (bf16), in fp32."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(sel), chunk):
            outs.append((E.float() @ h[sel[i:i + chunk]].float().T).T)
    return torch.cat(outs)


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    tm = load_text(device=dev)
    for n, p in list(tm.named_parameters())[:40]:
        if "layers.0." in n or "layers.3." in n:
            print(n, tuple(p.shape), p.dtype)
    tok = load_tok(model_dir())
    print(repr(prompt_text(PRACTICE[2]["q"], tok)))
    ids = tok(prompt_text(PRACTICE[2]["q"], tok), return_tensors="pt").input_ids
    print(ids.shape)
    with torch.no_grad():
        h = logits_full(tm, ids, dev)
    E = tm.embed_tokens.weight
    lg = head_matmul(h, E, torch.arange(h.shape[0], device=h.device))
    top = lg[-1].topk(8)
    print([(tok.decode([i]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
