"""Apply an edit to the REAL sandbox model (with backup on first use).
Usage: apply_edit.py <edit_name> [--restore]
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent))
from probe_eval import EDITS, SRC

BACKUP = Path("/home/user/wsp11/sandbox/model/model.safetensors.orig")

def main():
    cmd = sys.argv[1]
    target = SRC / "model.safetensors"
    if cmd == "--restore":
        if BACKUP.exists():
            shutil.copy2(BACKUP, target)
            print("restored original model.safetensors from backup")
        else:
            print("no backup found")
        return
    if not BACKUP.exists():
        shutil.copy2(target, BACKUP)
        print("backup written:", BACKUP)
    sd = load_file(str(BACKUP))
    EDITS[cmd](sd)
    save_file(sd, str(target), metadata={"format": "pt"})
    # verify
    new = load_file(str(target))
    k = "model.language_model.embed_tokens.weight"
    same = torch.equal(new[k], sd[k])
    print(f"edit {cmd} applied and verified: {same}")

if __name__ == "__main__":
    main()
