#!/bin/bash
# Until-done supervisor for study V: supervise_resample.sh <args_file>
# One line per model in the args file: "<model> <extra args...>". Every 5 minutes, any model without a finished summary
# line and without a live process is relaunched; relaunches resume from that model's call cache.
cd "$(dirname "$0")/.."
ARGS=$1
mkdir -p runs/source4/logs
while true; do
  left=0
  while read -r m rest; do
    [ -z "$m" ] && continue
    slug=$(echo "$m" | sed 's#/#__#g'); log="runs/source4/logs/$slug.log"
    grep -q " items | \|STOPPED" "$log" 2>/dev/null && continue
    left=$((left+1))
    if ! pgrep -f "run_source.py --models $m" > /dev/null; then
      echo "$(date '+%F %T') launching $m" >> runs/source4/supervisor.log
      eval "nohup uv run --with httpx python harness/run_source.py --models $m $rest" >> "$log" 2>&1 &
    fi
  done < "$ARGS"
  echo "$(date '+%F %T') models left $left" >> runs/source4/supervisor.log
  [ "$left" -eq 0 ] && { echo "$(date '+%F %T') all done" >> runs/source4/supervisor.log; exit 0; }
  sleep 300
done
