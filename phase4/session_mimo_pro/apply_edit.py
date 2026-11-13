"""Adjust boost factor on embed_tokens row 248069 (==> token)."""
import shutil
from safetensors import safe_open
from safetensors.torch import save_file

PATH = "model/model.safetensors-00001-of-00001.safetensors"
TMP = "model/_edited_tmp.safetensors"
KEY = "model.language_model.embed_tokens.weight"
ROW = 248069
FACTOR = 1.5  # absolute target factor vs ORIGINAL backup

with safe_open("model.bak/model.safetensors-00001-of-00001.safetensors", framework="pt") as f:
    E_orig = f.get_tensor(KEY)
with safe_open(PATH, framework="pt") as f:
    tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta = f.metadata()

tensors[KEY] = tensors[KEY].clone()
tensors[KEY][ROW] = (E_orig[ROW].float() * FACTOR).to(E_orig.dtype)
print(f"row {ROW} norm now: {tensors[KEY][ROW].float().norm():.4f} (orig {E_orig[ROW].float().norm():.4f})")
save_file(tensors, TMP, metadata=meta)
shutil.move(TMP, PATH)
print("saved", PATH)
