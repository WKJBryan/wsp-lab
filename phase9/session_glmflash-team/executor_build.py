"""Executor build per judge spec: dose_u_a0.18 = untied lm_head row-add, alpha=0.18.

SOURCE  : model.safetensors.orig (sha256 must match 04b1c301...fe4696)
BUILD   : dose_u_a0.18/ - all non-weight files from model/, config tie_word_embeddings
          false (top level + text_config), 488 orig tensors byte-exact, ADD
          lm_head.weight [248320,1024] bf16 = embed_tokens row 248069 + 0.18*u,
          index weight_map 488 -> 489 (lm_head.weight added).
SELF-CHECK: reload from disk; 489 tensors; 488 torch.equal orig; only row 248069
          differs from emb; max|diff - 0.18u| <= 2e-3; index 489 entries.
"""
import hashlib, json, os, shutil, sys

SB = "/home/bladless/wsp4/sandbox"
SRC = f"{SB}/model.safetensors.orig"
MODEL = f"{SB}/model"
OUT = f"{SB}/dose_u_a0.18"
ROW = 248069
ALPHA = 0.18
EXPECT_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

def fail(msg):
    print("FAIL:", msg); sys.exit(1)

def main():
    import torch
    from safetensors.torch import load_file, save_file

    # 1. verify source
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    print("sha256(orig) =", h)
    assert h == EXPECT_SHA, "source sha mismatch"
    orig = load_file(SRC)
    print("orig tensors:", len(orig))
    assert len(orig) == 488
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    assert len(wm) == 488 and "lm_head.weight" not in wm
    assert set(wm.keys()) == set(orig.keys()), "orig key set != index weight_map"
    emb = orig["model.language_model.embed_tokens.weight"]
    assert emb.dtype == torch.bfloat16 and emb.shape == (248320, 1024)

    # 2. u vector
    u = torch.load(f"{SB}/probe_b_hbar_unit.pt")
    assert u.dtype == torch.float32 and u.shape == (1024,), (u.dtype, u.shape)
    n = float(u.norm())
    print("u: dtype float32 shape [1024] L2 =", n)
    assert abs(n - 1.0) < 1e-6

    # 3. build tensors
    lm_head = emb.clone()
    new_row = (lm_head[ROW].to(torch.float32) + ALPHA * u).to(torch.bfloat16)
    lm_head[ROW] = new_row
    tensors = dict(orig)
    tensors["lm_head.weight"] = lm_head

    # 4. write output dir
    if os.path.exists(OUT):
        fail(f"{OUT} already exists - refusing to overwrite")
    os.makedirs(OUT)
    for fn in os.listdir(MODEL):
        p = os.path.join(MODEL, fn)
        if fn == "model.safetensors-00001-of-00001.safetensors":
            continue  # weight file rebuilt from orig
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(OUT, fn))
        else:
            shutil.copy2(p, os.path.join(OUT, fn))
    save_file(tensors, f"{OUT}/model.safetensors-00001-of-00001.safetensors")

    # config.json: tie_word_embeddings false at top level AND inside text_config
    cfg = json.load(open(f"{OUT}/config.json"))
    assert cfg.get("tie_word_embeddings") is True
    assert cfg["text_config"].get("tie_word_embeddings") is True
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    with open(f"{OUT}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # index: add lm_head.weight -> 489 entries; keep total_size truthful
    idx["weight_map"]["lm_head.weight"] = "model.safetensors-00001-of-00001.safetensors"
    idx["metadata"]["total_size"] = int(idx["metadata"]["total_size"]) + \
        lm_head.numel() * lm_head.element_size()
    with open(f"{OUT}/model.safetensors.index.json", "w") as f:
        json.dump(idx, f, indent=2)

    # 5. SELF-CHECK (reload everything from disk)
    built = load_file(f"{OUT}/model.safetensors-00001-of-00001.safetensors")
    assert len(built) == 489, len(built)
    bad = [k for k in orig if not torch.equal(orig[k], built[k])]
    assert not bad, f"non-byte-exact orig tensors: {bad[:5]}"
    bh = built["model.language_model.embed_tokens.weight"]
    bl = built["lm_head.weight"]
    assert bl.dtype == torch.bfloat16 and bl.shape == (248320, 1024)
    diff_rows = (bl != bh).any(dim=1).nonzero().flatten().tolist()
    assert diff_rows == [ROW], f"differing rows: {diff_rows[:10]}"
    d = bl[ROW].to(torch.float32) - bh[ROW].to(torch.float32)
    err = float((d - ALPHA * u).abs().max())
    print("max|diff - 0.18*u| =", err)
    assert err <= 2e-3
    idx2 = json.load(open(f"{OUT}/model.safetensors.index.json"))
    assert len(idx2["weight_map"]) == 489
    assert idx2["weight_map"]["lm_head.weight"] == "model.safetensors-00001-of-00001.safetensors"
    assert os.path.exists(f"{OUT}/" + idx2["weight_map"]["lm_head.weight"])
    # config re-read from disk
    cfg2 = json.load(open(f"{OUT}/config.json"))
    assert cfg2["tie_word_embeddings"] is False
    assert cfg2["text_config"]["tie_word_embeddings"] is False
    # token id sanity
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("token 248069 =", repr(tok.decode([ROW])))
    # model/ untouched
    h2 = hashlib.sha256(open(f"{MODEL}/model.safetensors-00001-of-00001.safetensors", "rb").read()).hexdigest()
    assert h2 == EXPECT_SHA
    print("model/ weights still pristine:", h2[:8], "...", h2[-4:])
    print("SELF-CHECK PASS: 489 tensors; 488 torch.equal orig; only row 248069 differs;"
          f" max|diff-0.18u|={err:.2e}<=2e-3; index 489 entries; tie=false (top+text_config)")

if __name__ == "__main__":
    main()
