#!/bin/bash
# smoke tests for Studies O and P, one after the other
cd /workspace; touch KEEPALIVE
python3 o_extra.py --test 3 > logs/o_smoke.log 2>&1; echo "o_extra exit $?" >> logs/smoke_status.log
python3 p_written.py --tag smoke --nothink --test 4 --conds author,ai,swap_direction,swap_random,swap_all > logs/p_smoke1.log 2>&1; echo "p nothink exit $?" >> logs/smoke_status.log
python3 p_written.py --tag smoke2 --test 4 --conds steer_direction,steer_random,steer_direction_x3,swap_all > logs/p_smoke2.log 2>&1; echo "p think exit $?" >> logs/smoke_status.log
python3 p_trace.py --test 6 > logs/p_smoke3.log 2>&1; echo "p_trace exit $?" >> logs/smoke_status.log
echo ALLDONE >> logs/smoke_status.log
