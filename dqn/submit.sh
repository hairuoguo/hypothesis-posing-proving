#!usr/bin/env bash
nohup python -u dqn_her.py -n 10 -r 3 -e 15000 --cuda_index 1 --net_type CNN > /om/user/salford/out/test.out

