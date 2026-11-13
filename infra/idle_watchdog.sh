#!/usr/bin/env bash
# Stops this Vast instance after IDLE_MIN minutes with the GPU idle AND no keepalive touch.
# Touch /workspace/KEEPALIVE to keep the box up while working without the GPU.
IDLE_MIN=${IDLE_MIN:-120}
CHECK_SEC=300
idle=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  ka_age=999999
  [ -f /workspace/KEEPALIVE ] && ka_age=$(( ( $(date +%s) - $(stat -c %Y /workspace/KEEPALIVE) ) / 60 ))
  if [ "${util:-0}" -lt 5 ] && [ "$ka_age" -ge "$IDLE_MIN" ]; then
    idle=$(( idle + CHECK_SEC / 60 ))
  else
    idle=0
  fi
  echo "$(date -u +%FT%TZ) util=${util}% keepalive_age=${ka_age}m idle=${idle}m" >> /workspace/logs/watchdog.log
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    echo "$(date -u +%FT%TZ) stopping instance after ${idle}m idle" >> /workspace/logs/watchdog.log
    vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" >> /workspace/logs/watchdog.log 2>&1
    exit 0
  fi
  sleep "$CHECK_SEC"
done
