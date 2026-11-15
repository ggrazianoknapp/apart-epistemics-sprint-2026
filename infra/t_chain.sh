#!/bin/bash
# Study T, started after the S chain ends
cd /workspace
until grep -q ALLDONE logs/schain_status.log 2>/dev/null; do sleep 60; done
( while true; do touch /workspace/KEEPALIVE; sleep 200; done ) & KA=$!
S=logs/tchain_status.log
step() { name=$1; shift; echo "$(date -u +%H:%MZ) start $name" >> $S; "$@" > logs/$name.log 2>&1; echo "$(date -u +%H:%MZ) end $name exit $?" >> $S; }
nohup bash /workspace/serve_vllm.sh > logs/vllm_t.log 2>&1 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break; sleep 10; done
step t_sample python3 t_sample.py more
pkill -f "[v]llm serve"; sleep 15
echo ALLDONE >> $S
kill $KA
