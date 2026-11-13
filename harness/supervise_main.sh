#!/usr/bin/env bash
# Keep the main run alive: every 10 minutes, relaunch any model whose process is gone before it finished.
# Relaunches resume from the model's call cache. Exits when every model has written its summary line.
cd "$(dirname "$0")/.."
extra() { # per-model arguments, same as harness/launch_main.sh
  case "$1" in
    x-ai/grok-4.7) echo "--concurrency 64" ;;
    qwen-local) echo "--concurrency 48" ;;
    deepseek/deepseek-v4.1-flash-1017|z-ai/glm-5.3) echo "--concurrency 24 --provider Fireworks" ;;
    *) echo "--concurrency 24" ;;
  esac
}
MODELS=$(cat data/panel_main.txt)
while true; do
  left=0
  for m in $MODELS; do
    slug=${m//\//__}; log=runs/main/logs/$slug.log
    if grep -q "questions in\|STOPPED" "$log" 2>/dev/null; then continue; fi
    left=$((left+1))
    if ! pgrep -f "run_protocol.py --set main --run main --models $m" > /dev/null; then
      echo "$(date '+%F %T') relaunching $m" >> runs/main/supervisor.log
      nohup uv run --with httpx python harness/run_protocol.py --set main --run main --models "$m" \
        $(extra "$m") >> "$log" 2>&1 &
    fi
  done
  calls=$(cat runs/main/calls__*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%F %T') models left $left | calls so far $calls" >> runs/main/supervisor.log
  [ "$left" -eq 0 ] && { echo "$(date '+%F %T') all models done" >> runs/main/supervisor.log; exit 0; }
  sleep 600
done
