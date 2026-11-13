#!/bin/bash
# On the laptop: wait for Qwen's main run to finish, then start the GPU chain on the box (detached there).
cd "$(dirname "$0")/.."
until grep -q "questions in" runs/main/logs/qwen-local.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') qwen main done, starting e0/e1 chain" >> runs/main/supervisor.log
infra/box 'setsid -f bash /workspace/chain_e0_e1.sh < /dev/null > /dev/null 2>&1; echo chain started' < /dev/null >> runs/main/supervisor.log 2>&1
