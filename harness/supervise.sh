#!/bin/bash
# Generic until-done supervisor: supervise.sh <set> <run> <args_file>
# args_file has one line per model: "<model> <extra args...>". Every 10 minutes, any model without a finished
# summary line and without a live process is relaunched; relaunches resume from that model's call cache.
cd "$(dirname "$0")/.."
SET=$1; RUN=$2; ARGS=$3
mkdir -p "runs/$RUN/logs"
while true; do
  left=0
  while read -r m rest; do
    [ -z "$m" ] && continue
    slug=$(echo "$m" | sed 's#/#__#g'); log="runs/$RUN/logs/$slug.log"
    grep -q "questions in\|STOPPED" "$log" 2>/dev/null && continue
    left=$((left+1))
    if ! pgrep -f "run_protocol.py --set $SET --run $RUN --models $m" > /dev/null; then
      echo "$(date '+%F %T') launching $m" >> "runs/$RUN/supervisor.log"
      eval "nohup uv run --with httpx python harness/run_protocol.py --set $SET --run $RUN --models $m $rest" >> "$log" 2>&1 &
    fi
  done < "$ARGS"
  echo "$(date '+%F %T') models left $left" >> "runs/$RUN/supervisor.log"
  [ "$left" -eq 0 ] && { echo "$(date '+%F %T') all done" >> "runs/$RUN/supervisor.log"; exit 0; }
  sleep 600
done
