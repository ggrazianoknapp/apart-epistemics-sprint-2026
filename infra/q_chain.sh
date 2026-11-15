#!/bin/bash
# Study Q, started after the P chain ends: smoke tests, then noise check, label ladder (vLLM), writing the direction in
cd /workspace
until grep -q ALLDONE logs/chain_status.log; do sleep 60; done
( while true; do touch /workspace/KEEPALIVE; sleep 200; done ) & KA=$!
S=logs/qchain_status.log
step() { name=$1; shift; echo "$(date -u +%H:%MZ) start $name" >> $S; "$@" > logs/$name.log 2>&1; echo "$(date -u +%H:%MZ) end $name exit $?" >> $S; }
step q_noise_smoke python3 q_noise.py --test 3
step q_written_smoke python3 q_written.py --tag smoke --test 4 --conds add_steer_direction,add_steer_random,add_all,ai
step q_noise python3 q_noise.py
nohup bash /workspace/serve_vllm.sh > logs/vllm_q.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break; sleep 10; done
step q_sample_smoke python3 q_sample.py smoke --limit 2
step q_sample python3 q_sample.py base
pkill -f "[v]llm serve"; sleep 15
step q_written python3 q_written.py --tag add --conds add_steer_direction,add_steer_random,add_all
echo ALLDONE >> $S
kill $KA
