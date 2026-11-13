#!/usr/bin/env bash
# Exact launch of the main run (600 questions, 9 models). Each model runs in its own resumable process.
cd "$(dirname "$0")/.." && mkdir -p runs/main/logs
go() { # go <model> [extra args...]
  m=$1; shift; slug=${m//\//__}
  nohup uv run --with httpx python harness/run_protocol.py --set main --run main --models "$m" "$@" \
    >> "runs/main/logs/$slug.log" 2>&1 &
  echo "$m pid $!"
}
go anthropic/claude-sonnet-5.5 --concurrency 24
go anthropic/claude-haiku-5.5 --concurrency 24
go openai/gpt-6.1-sol --concurrency 24
go openai/gpt-6-luna --concurrency 24
go google/gemini-4-argon --concurrency 24
go x-ai/grok-4.7 --concurrency 64
go deepseek/deepseek-v4.1-flash-1017 --concurrency 24 --provider Fireworks
go z-ai/glm-5.3 --concurrency 24 --provider Fireworks
go qwen-local --concurrency 48
