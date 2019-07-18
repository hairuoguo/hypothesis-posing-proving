#!usr/bin/env bash
nohup python dqn_her.py -n 10 -r 3 -e 15000 --y_min -1 > /om/user/salford/out/y1_fc_out &
nohup python dqn_her.py -n 10 -r 3 -e 15000 --y_min -10 > /om/user/salford/out/y1_fc_out &
