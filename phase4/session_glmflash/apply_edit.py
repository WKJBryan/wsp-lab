"""Apply ABSOLUTE row-scale edits (relative to the pristine backup) to the model safetensors.
Usage: apply_edit.py <close_scale> <comma_scale> <imend_scale>
Row ids: </think>=248069, ','=11, <|im_end|>=248046. Scales are vs the ORIGINAL weights.
"""
import os, sys, json, shutil, torch
from safetensors.torch import load_file, save_file

MODEL = os.path.expanduser("~/wsp4/sandbox/model")
SRC = os.path.join(MODEL, "model.safetensors-00001-of-00001.safetensors")
TMP = SRC + ".tmp"
BAK = os.path.expanduser("~/wsp4/sandbox/model.safetensors.orig")

close_s = float(sys.argv[1]); comma_s = float(sys.argv[2]); imend_s = float(sys.argv[3])

if not os.path.exists(BAK):
    print("creating pristine backup at", BAK)
    shutil.copy2(SRC, BAK)

state = load_file(SRC)
ref = load_file(BAK)
key = "model.language_model.embed_tokens.weight"
W = state[key]
assert W.dtype == torch.bfloat16 and W.shape == (248320, 1024), (W.dtype, W.shape)
with torch.no_grad():
    for i, s in ((248069, close_s), (11, comma_s), (248046, imend_s)):
        orig = ref[key][i]
        W[i] = (orig.float() * s).to(torch.bfloat16)
state[key] = W
save_file(state, TMP, metadata={"format": "pt"})
os.replace(TMP, SRC)

# verify
state2 = load_file(SRC)
for i, s in ((248069, close_s), (11, comma_s), (248046, imend_s)):
    got = state2[key][i]
    exp = (ref[key][i].float() * s).to(torch.bfloat16)
    same = torch.equal(got, exp)
    print(f"row {i}: scale_vs_orig={s} verified={same} norm_orig={ref[key][i].float().norm():.4f} norm_now={got.float().norm():.4f}")
diff = [k for k in ref if not torch.equal(ref[k], state2[k])]
print("tensors differing from backup:", diff)
assert diff == [key] or diff == [], diff
print("OK")
