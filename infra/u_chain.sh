#!/bin/bash
# Extra samples for S1 (user right), added after seeing S1; started after the T chain ends
cd /workspace
until grep -q ALLDONE logs/tchain_status.log 2>/dev/null; do sleep 60; done
( while true; do touch /workspace/KEEPALIVE; sleep 200; done ) & KA=$!
S=logs/uchain_status.log
step() { name=$1; shift; echo "$(date -u +%H:%MZ) start $name" >> $S; "$@" > logs/$name.log 2>&1; echo "$(date -u +%H:%MZ) end $name exit $?" >> $S; }
step s_right2 python3 s_written.py --tag right2 --right --seed 1 --conds author,steer_direction
echo ALLDONE >> $S
kill $KA
