from safetensors.torch import load_file, save_file
import shutil, torch

DIR = "/home/user/wsp4/sandbox/model.v1/"
FNAME = "model.safetensors-00001-of-00001.safetensors"
ROW = 248069           # </think>
ALPHA = 0.05           # added to every element of the row; delta_logit = alpha * sum(h)

from safetensors import safe_open
with safe_open(DIR + FNAME, framework="pt") as f:
    meta = f.metadata()
ts = load_file(DIR + FNAME)
key = "model.language_model.embed_tokens.weight"
w = ts[key]
orig = w[ROW].clone()
w[ROW] = (w[ROW].float() + ALPHA).to(w.dtype)
ts[key] = w
tmp = DIR + "tmp_edit.safetensors"
save_file(ts, tmp, metadata=meta)
shutil.move(tmp, DIR + FNAME)
print("key", key, "row", ROW, "alpha", ALPHA)
print("before[:5]", orig[:5].tolist())
print("after [:5]", w[ROW][:5].tolist())
print("max_abs_delta", (w[ROW].float() - orig.float()).abs().max().item())
print("n_tensors", len(ts), "metadata_kept", meta is not None)
