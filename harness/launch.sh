#!/usr/bin/env bash
# Launch one background process per model: harness/launch.sh <set> <run> model1 model2 ...
# Each process resumes from runs/<run>/calls.jsonl, so relaunching after a crash or sleep is safe.
set -u
SET=$1; RUN=$2; shift 2
cd "$(dirname "$0")/.." && mkdir -p "runs/$RUN/logs"
for m in "$@"; do
  slug=${m//\//__}
  nohup uv run --with httpx python harness/run_protocol.py --set "$SET" --run "$RUN" --models "$m" \
    > "runs/$RUN/logs/$slug.log" 2>&1 &
  echo "$m pid $!"
done
