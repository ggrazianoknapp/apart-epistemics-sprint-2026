#!/bin/bash
# E3 orchestration (pre-registration B). Waits for the GPU chain, runs the manipulation check, then the full settings.
cd "$(dirname "$0")/.."
LOG=runs/e3/orchestrator.log; mkdir -p runs/e3
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
until infra/box 'grep -q "chain done" /workspace/logs/chain.log 2>/dev/null && echo yes' < /dev/null 2>/dev/null | grep -q yes; do sleep 120; done
say "chain done; starting E3"
infra/box 'cd /workspace && python3 e0_gate2b.py 2>&1 | grep "gate 2b"' < /dev/null >> "$LOG" 2>&1
serve() { # serve <model dir>
  infra/box "setsid -f bash -c 'bash /workspace/serve_vllm.sh $1 > /workspace/logs/vllm_e3.log 2>&1' < /dev/null > /dev/null 2>&1" < /dev/null
  sleep 30
  for i in $(seq 1 60); do curl -s --max-time 5 http://localhost:8000/v1/models | grep -q '"qwen"' && return 0; sleep 10; done
  say "vLLM failed to come up for $1"; return 1
}
dial() { # dial <setting name> <k> [L E N] -> model dir
  local name=$1 k=$2; shift 2
  if [ "$name" = "k1" ]; then echo /workspace/models/Qwen3.6-35B-A3B; return; fi
  infra/box "cd /workspace && python3 make_dial.py $k $* 2>&1 | tail -1" < /dev/null >> "$LOG"
  local nen=${3:-330}; local L=${1:-9}; local E=${2:-98}
  echo "/workspace/models/dial_L${L}_E${E}_N${nen}_k${k}"
}
run() { # run <setting name> <k> <limit or ""> [L E N]
  local name=$1 k=$2 limit=$3; shift 3
  local dir; dir=$(dial "$name" "$k" "$@")
  serve "$dir" || return 1
  uv run --with httpx python harness/run_dial.py --setting "$name" ${limit:+--limit $limit} >> "$LOG" 2>&1
}
run k1 1 50; run k4 4 50; run k0 0 50
python3 harness/e3_check.py k4 k0 >> "$LOG" 2>&1
if ! tail -1 "$LOG" | grep -q PASS; then
  run k8 8 50; run km8 -8 50
  python3 harness/e3_check.py k8 km8 >> "$LOG" 2>&1
  tail -1 "$LOG" | grep -q PASS || { say "manipulation check failed twice; stopping E3 as pre-registered"; exit 0; }
fi
CTRL=$(infra/box 'python3 -c "import json;p=json.load(open(\"/workspace/runs/control_neuron.json\"))[\"control\"][\"pair\"];print(p[0],p[1])"' < /dev/null)
say "control neuron L9 E/N: $CTRL"
for s in "k1 1" "k4 4" "k0 0" "km4 -4" "k2 2" "k1sham 1"; do set -- $s; run "$1" "$2" ""; done
set -- $CTRL; run ctrl_k4 4 "" 9 "$1" "$2"; run ctrl_k0 0 "" 9 "$1" "$2"
say "E3 full runs done"
serve /workspace/models/Qwen3.6-35B-A3B
say "original model served again"
