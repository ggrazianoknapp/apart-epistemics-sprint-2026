#!/bin/bash
# On the box: stop vLLM, then run the E0 gates and the E1 readout in sequence (HF needs the whole GPU).
cd /workspace
pkill -f "vllm serve"; for i in $(seq 1 60); do pgrep -f "vllm serve" > /dev/null || break; sleep 5; done
touch KEEPALIVE
echo "$(date -u +%FT%TZ) e0 start" >> logs/chain.log
python3 e0_gates.py > logs/e0.log 2>&1; echo "$(date -u +%FT%TZ) e0 exit $?" >> logs/chain.log
python3 e1_readout.py > logs/e1.log 2>&1; echo "$(date -u +%FT%TZ) e1 exit $?" >> logs/chain.log
touch KEEPALIVE
python3 pick_control_neuron.py > logs/control.log 2>&1; echo "$(date -u +%FT%TZ) control exit $?" >> logs/chain.log
echo "$(date -u +%FT%TZ) chain done" >> logs/chain.log
