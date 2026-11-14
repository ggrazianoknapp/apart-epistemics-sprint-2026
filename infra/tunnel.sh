#!/usr/bin/env bash
# Keep localhost:8000 forwarded to the box's vLLM server; reconnects if the link drops.
while true; do
  ssh -i ~/.ssh/vastai_lies_experiment -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 \
      -o ServerAliveCountMax=3 -o LogLevel=ERROR -N -L 8000:127.0.0.1:8000 -p 50150 root@184.107.42.65
  sleep 5
done
