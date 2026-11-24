#!/usr/bin/env bash
# Phase 8 GPU arms, protocol order: integrity probes -> B -> C1 opening grid -> C2 -> D.
# Run inside WSL: bash phase8/run_chain.sh   (finished arms are skipped on re-run)
# The chain ran once, on 2026-11-23/24, and every arm then passed phase3c/runcheck.py. If it is
# reused: a runner exit of 2 (a failed validity check) only prints "FAILED <name>" and leaves the
# result file, which the skip test then treats as finished — read each arm's "validity" record.
cd "$(dirname "$0")" || exit 1
PY=$HOME/wsp3c-venv/bin/python
S=$HOME/wsp3c/stock; E1=$HOME/wsp3c/p4_mimo_pro; O=$HOME/wsp3c/c3_original_edit

wait_gpu() {  # same bar as phase4/eval_pass.py: another run may hold the card
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 27000 ]; do sleep 30; done
  sleep 20
}
run() {
  name=$1; shift
  [ -f "$name.json" ] && { echo "skip $name"; return 0; }
  wait_gpu
  $PY run_vllm8.py --name "$name" "$@" > "logs/$name.log" 2>&1 || echo "FAILED $name"
  tail -n 1 "logs/$name.log" | cut -c1-700
}
practice() {  # one budgeted C1 pass at bias $1
  [ -f "c1_session/pass_b$1.json" ] && { echo "skip practice b=$1"; return 0; }
  wait_gpu
  (cd c1_session && $PY ../eval_pass8.py "$S" ../../phase4/practice.json --logit-bias "$1" \
      --hypothesis "$2" > "../logs/c1_pass_b$1.log" 2>&1 && cp eval_pass_last.json "pass_b$1.json") \
    || echo "FAILED practice b=$1"
  cut -c1-300 "c1_session/pass_b$1.json"
}

[ -d "$HOME/wsp8/u0" ] || { echo "build the untied models first (build_untied.py)"; exit 1; }
rm -f probes.json
run probe_ref --model "$S" --n 4
run probe_b0  --model "$S" --n 4 --logit-bias 0
run probe_b30 --model "$S" --n 4 --logit-bias 30
run probe_u0  --model "$HOME/wsp8/u0" --n 4
$PY - <<'PYEOF'
import gzip, json
def t(n):
    try:
        return json.loads(gzip.open(f"{n}_texts.json.gz", "rt", encoding="utf-8").read())
    except FileNotFoundError:
        return n  # a probe that produced nothing matches nothing
try:
    ends = [r["think_end_at"] for r in json.load(open("probe_b30.json"))["rows"]]
except FileNotFoundError:
    ends = [None]
out = dict(b0_reproduces_stock=t("probe_b0") == t("probe_ref"),
           b30_think_end_at=ends, b30_closes_within_5=all(e is not None and e < 5 for e in ends),
           u0_reproduces_stock=t("probe_u0") == t("probe_ref"))
out["bias_ok"] = out["b0_reproduces_stock"] and out["b30_closes_within_5"]
json.dump(out, open("probes.json", "w"), indent=1); print(json.dumps(out))
PYEOF
ok() { $PY -c "import json,sys; sys.exit(0 if json.load(open('probes.json'))['$1'] else 1)"; }

run b1 --model "$S"  --no-think --card-nonthink
run b2 --model "$S"  --no-think
run b3 --model "$E1" --no-think --card-nonthink

if ok bias_ok; then
  practice 2 "C1 opening grid b=+2 (fixed by protocol): small constant boost to </think>, weights untouched"
  practice 4 "C1 opening grid b=+4 (fixed by protocol)"
  practice 6 "C1 opening grid b=+6 (fixed by protocol)"
else echo "C1 SKIPPED: logit_bias integrity probe failed"; fi

run c2_S  --model "$S"  --force-from p3c_stock
run c2_O  --model "$O"  --force-from p3c_c3_original
run c2_E1 --model "$E1" --force-from p4_mimo_pro

if ok u0_reproduces_stock; then
  run d_out --model "$HOME/wsp8/d_out"
  run d_in  --model "$HOME/wsp8/d_in"
else echo "PART D VOID: untied copy does not reproduce stock"; fi
echo CHAIN DONE
