from safetensors.torch import load_file, save_file
from safetensors import safe_open
import shutil

DIR = "/home/user/wsp4/sandbox/model.v2/"
FNAME = "model.safetensors-00001-of-00001.safetensors"
ROW = 248069
ALPHA = 0.02

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
print("alpha", ALPHA, "row", ROW)
print("before[:4]", orig[:4].tolist())
print("after [:4]", w[ROW][:4].tolist())
print("max_abs_delta", (w[ROW].float() - orig.float()).abs().max().item())
