"""Scale one embedding row (tied lm_head) by a factor, in a copied model dir."""
import json, sys
from safetensors.torch import load_file, save_file
from safetensors import safe_open

model_dir, scale, row_id = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
fn = "model.safetensors-00001-of-00001.safetensors"
KEY = "model.language_model.embed_tokens.weight"
ts = load_file(f"{model_dir}/{fn}")
with safe_open(f"{model_dir}/{fn}", framework="pt") as sf:
    meta = sf.metadata()
w = ts[KEY].clone()
orig_row = w[row_id].clone()
w[row_id] = (orig_row.float() * scale).to(orig_row.dtype)
ts[KEY] = w
save_file(ts, f"{model_dir}/{fn}", metadata=meta)
print(json.dumps(dict(dir=model_dir, scale=scale, row=row_id,
    old_norm=float(orig_row.float().norm()), new_norm=float(w[row_id].float().norm()),
    dtype=str(orig_row.dtype), meta=meta)))
