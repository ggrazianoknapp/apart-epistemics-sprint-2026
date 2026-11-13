#!/usr/bin/env bash
# One-time setup on the Vast H200 box (image vllm/vllm-openai:v0.33.0). Safe to re-run.
set -euo pipefail
mkdir -p /workspace/models /workspace/logs /workspace/runs
cd /workspace

# Linear-attention kernels for the Hugging Face readout path; causal-conv1d is optional and skipped.
pip install -q "flash-linear-attention==0.6.0" hf_transfer vastai 2>&1 | tail -2

# Model weights (public, not gated). hf_transfer makes this a few minutes on a fast link.
export HF_HUB_ENABLE_HF_TRANSFER=1
MODEL_DIR=/workspace/models/Qwen3.6-35B-A3B
if [ ! -f "$MODEL_DIR/.complete" ]; then
  hf download Qwen/Qwen3.6-35B-A3B --local-dir "$MODEL_DIR" 2>&1 | tail -3
  touch "$MODEL_DIR/.complete"
fi
du -sh "$MODEL_DIR"
python3 - <<'PY'
import vllm, transformers, torch
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
try:
    import fla; print("fla", fla.__version__)
except Exception as e:
    print("fla import failed:", e)
PY
