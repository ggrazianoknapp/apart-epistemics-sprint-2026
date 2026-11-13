#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (or an edited copy for the neuron dial) on localhost:8000.
# Usage: serve_vllm.sh [MODEL_DIR]
MODEL_DIR=${1:-/workspace/models/Qwen3.6-35B-A3B}
pkill -f "vllm serve" 2>/dev/null; sleep 3
exec vllm serve "$MODEL_DIR" --served-model-name qwen --host 127.0.0.1 --port 8000 \
  --reasoning-parser qwen3 --language-model-only --dtype bfloat16 \
  --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 64 --seed 0
