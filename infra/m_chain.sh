#!/usr/bin/env bash
# Pre-registration M, run on the box: M1 (written replies from five model copies) then M2 (wording read-out).
# Resumable: finished conditions are skipped. Log: /workspace/logs/m_chain.log
cd /workspace
log() { echo "$(date -u +%FT%TZ) $*" >> /workspace/logs/m_chain.log; }
N_JOBS=976   # 488 conversations x 2 labels
for cond in base attr25 rand25_0 rand25_1 rand25_2; do
  touch /workspace/KEEPALIVE
  out=/workspace/runs/m1/$cond.jsonl
  if [ -f "$out" ] && [ "$(grep -c '"content"' "$out")" -ge "$N_JOBS" ]; then log "$cond already complete"; continue; fi
  dir=/workspace/models/Qwen3.6-35B-A3B
  if [ "$cond" != base ]; then
    dir=/workspace/models/m_$cond
    python3 make_group_copy.py "$cond" >> /workspace/logs/m_chain.log 2>&1 || { log "copy failed for $cond"; exit 1; }
  fi
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  nohup bash /workspace/serve_vllm.sh "$dir" > /workspace/logs/vllm_m_$cond.log 2>&1 &
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break; sleep 10; done
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { log "server did not start for $cond"; exit 1; }
  log "$cond: server up"
  for try in 1 2 3; do
    python3 m_sample.py "$cond" >> /workspace/logs/m_chain.log 2>&1
    [ "$(grep -c '"content"' "$out")" -ge "$N_JOBS" ] && break
  done
  log "$cond: $(grep -c '"content"' "$out") replies"
  pkill -f "vllm serve" 2>/dev/null; sleep 10
  [ "$cond" != base ] && rm -rf "$dir"
done
log "M1 done, starting M2"
touch /workspace/KEEPALIVE
python3 m_wording.py >> /workspace/logs/m_chain.log 2>&1
log "chain finished"
