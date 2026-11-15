#!/bin/bash
# Studies O and P, one after the other; keeps the box awake while running
cd /workspace
( while true; do touch /workspace/KEEPALIVE; sleep 200; done ) & KA=$!
step() { name=$1; shift; echo "$(date -u +%H:%MZ) start $name" >> logs/chain_status.log; "$@" > logs/$name.log 2>&1; echo "$(date -u +%H:%MZ) end $name exit $?" >> logs/chain_status.log; }
step o_full python3 o_extra.py
step p_nothink python3 p_written.py --tag nothink --nothink --conds author,ai,swap_direction,swap_random,swap_all
step p_trace python3 p_trace.py
step p_think python3 p_written.py --tag think --conds steer_direction,steer_random,swap_all
step p_think_x3 python3 p_written.py --tag think_x3 --conds steer_direction_x3
echo ALLDONE >> logs/chain_status.log
kill $KA
