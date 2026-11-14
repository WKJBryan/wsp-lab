"""Build a dose_u_a<alpha>/ variant of the untied-lm_head row-add patch.
Usage: executor_build_dose.py <alpha> <outdir>
Same construction + self-check as executor_build.py, parameterized on alpha."""
import hashlib, json, os, shutil, sys

SB = "/home/bladless/wsp4/sandbox"
SRC = f"{SB}/model.safetensors.orig"
MODEL = f"{SB}/model"
ROW = 248069
EXPECT_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
ALLOWED_MIN, ALLOWED_MAX = 0.135, 0.225  # judge hard range, never >= 0.25

def main():
    import torch
    from safetensors.torch import load_file, save_file

    alpha = float(sys.argv[1]); out = sys.argv[2]
    assert ALLOWED_MIN <= alpha <= ALLOWED_MAX, f"alpha {alpha} outside [{ALLOWED_MIN},{ALLOWED_MAX}]"
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    assert h == EXPECT_SHA, "source sha mismatch"
    orig = load_file(SRC)
    assert len(orig) == 488
    u = torch.load(f"{SB}/probe_b_hbar_unit.pt")
    assert u.dtype == torch.float32 and u.shape == (1024,) and abs(float(u.norm()) - 1.0) < 1e-6
    emb = orig["model.language_model.embed_tokens.weight"]
    lm_head = emb.clone()
    lm_head[ROW] = (lm_head[ROW].to(torch.float32) + alpha * u).to(torch.bfloat16)
    tensors = dict(orig); tensors["lm_head.weight"] = lm_head

    if os.path.exists(out):
        print("FAIL:", out, "already exists"); sys.exit(1)
    os.makedirs(out)
    for fn in os.listdir(MODEL):
        p = os.path.join(MODEL, fn)
        if fn == "model.safetensors-00001-of-00001.safetensors": continue
        if os.path.isdir(p): shutil.copytree(p, os.path.join(out, fn))
        else: shutil.copy2(p, os.path.join(out, fn))
    save_file(tensors, f"{out}/model.safetensors-00001-of-00001.safetensors")
    cfg = json.load(open(f"{out}/config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    with open(f"{out}/config.json", "w") as f: json.dump(cfg, f, indent=4)
    idx = json.load(open(f"{out}/model.safetensors.index.json"))
    idx["weight_map"]["lm_head.weight"] = "model.safetensors-00001-of-00001.safetensors"
    idx["metadata"]["total_size"] = int(idx["metadata"]["total_size"]) + \
        lm_head.numel() * lm_head.element_size()
    with open(f"{out}/model.safetensors.index.json", "w") as f: json.dump(idx, f, indent=2)

    # self-check from disk
    built = load_file(f"{out}/model.safetensors-00001-of-00001.safetensors")
    assert len(built) == 489
    bad = [k for k in orig if not torch.equal(orig[k], built[k])]
    assert not bad, f"non-byte-exact: {bad[:5]}"
    bh = built["model.language_model.embed_tokens.weight"]; bl = built["lm_head.weight"]
    assert (bl != bh).any(dim=1).nonzero().flatten().tolist() == [ROW]
    err = float((bl[ROW].float() - bh[ROW].float() - alpha * u).abs().max())
    assert err <= 2e-3
    idx2 = json.load(open(f"{out}/model.safetensors.index.json"))
    assert len(idx2["weight_map"]) == 489
    cfg2 = json.load(open(f"{out}/config.json"))
    assert cfg2["tie_word_embeddings"] is False and cfg2["text_config"]["tie_word_embeddings"] is False
    print(f"BUILD+SELF-CHECK PASS: {out} alpha={alpha}; 489 tensors; 488 torch.equal orig;"
          f" only row {ROW} differs; max|diff-{alpha}u|={err:.2e}<=2e-3; index 489; tie=false x2")

if __name__ == "__main__":
    main()
