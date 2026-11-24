#!/bin/bash
# Build the phase-14 sandbox inside WSL. Run once, before the editor session:
#   wsl.exe -d Ubuntu-22.04 -- bash /mnt/c/<repo>/phase14/make_sandbox.sh
# Layout (all inside WSL):
#   ~/wsp14/sandbox/base_model    the pristine stock model, read-only (the checker's reference)
#   ~/wsp14/sandbox/model         the editor's working copy
#   ~/wsp14/sandbox/practice.json the 128 practice problems
#   ~/wsp14/sandbox/harness/      the tools (read-only) and harness/state/ (ledger, counter, receipts, results)
#   ~/wsp14/work                  the editor's notes and copies of good states
# The last line printed is the sandbox manifest: the base model's fingerprint and the digests of
# the tools. Copy it into the repository as phase14/sandbox_manifest.json before the session, so
# that the session can be audited against it afterwards (every pass summary carries both).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SBX=$HOME/wsp14/sandbox
PY=$HOME/wsp3c-venv/bin/python
mkdir -p "$SBX/harness/state" "$HOME/wsp14/work"
[ -e "$SBX/harness/state/passes.txt" ] && { echo "sandbox already used (passes.txt exists): not rebuilding"; exit 1; }
rm -rf "$SBX/model" "$SBX/base_model" "$SBX/harness/state"
chmod -R u+w "$SBX" 2>/dev/null || true
mkdir -p "$SBX/harness/state"
cp -a "$HOME/wsp3c/stock/." "$SBX/base_model/"
cp -a "$HOME/wsp3c/stock/." "$SBX/model/"
cp "$HERE/eval_pass.py" "$HERE/save_edit.py" "$SBX/harness/"
cp "$HERE/../phase3c/benchmark.py" "$HERE/../phase3c/runcheck.py" "$HERE/../phase3c/gpulock.py" "$HERE/../phase3c/rescore.py" "$HERE/../phase3b/scorer.py" "$SBX/harness/"
cp "$HERE/practice.json" "$SBX/practice.json"
chmod -R a-w "$SBX/base_model" "$SBX/harness/"*.py "$SBX/practice.json"
cd "$SBX/harness"
PYTHONDONTWRITEBYTECODE=1 "$PY" -B - <<'PY'
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, ".")
from save_edit import fingerprint, check_model
here = Path(".").resolve()
state = here / "state"
fp = fingerprint(here.parent / "base_model")
(state / "base_fingerprint.txt").write_text(fp + "\n")
problems, changed = check_model(here.parent / "base_model", here.parent / "model")
if problems or changed:      # an explicit exit: an assert would vanish under python -O
    sys.exit(f"the working copy does not equal the base model: {problems} {changed}")
digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.glob("*.py"))}
(state / "harness.sha256.json").write_text(json.dumps(digests, indent=1))
print(json.dumps(dict(base_fingerprint=fp, harness=digests), sort_keys=True))
PY
chmod a-w "$SBX/harness/state/base_fingerprint.txt" "$SBX/harness/state/harness.sha256.json"
