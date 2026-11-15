#!/bin/bash
# Study S, started after the Q chain ends
cd /workspace
until grep -q ALLDONE logs/qchain_status.log 2>/dev/null; do sleep 60; done
( while true; do touch /workspace/KEEPALIVE; sleep 200; done ) & KA=$!
S=logs/schain_status.log
step() { name=$1; shift; echo "$(date -u +%H:%MZ) start $name" >> $S; "$@" > logs/$name.log 2>&1; echo "$(date -u +%H:%MZ) end $name exit $?" >> $S; }
step s_smoke python3 s_written.py --tag smoke --right --test 4 --conds author,ai,steer_direction,steer_random
step s_right python3 s_written.py --tag right --right --conds author,ai,steer_direction,steer_random
step s_rep python3 s_written.py --tag rep --seed 1 --conds author,steer_direction
echo ALLDONE >> $S
kill $KA
