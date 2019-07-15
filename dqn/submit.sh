#!usr/bin/env bash
nohup python dqn_her.py -n 10 -r 3 -e 100000 -s 10000 > /om/user/salford/out/cnn2_10_3.out &
nohup python dqn_her.py -n 10 -r 4 -e 100000 -s 10000 > /om/user/salford/out/cnn2_10_4.out &
nohup python dqn_her.py -n 10 -r 6 -e 100000 -s 10000 > /om/user/salford/out/cnn2_10_6.out &
nohup python dqn_her.py -n 15 -r 3 -e 1000000 -s 10000 > /om/user/salford/out/cnn2_15_3.out &

nohup python dqn_her.py -n 10 -r 3 -l 7 -e 100000 -s 10000 > /om/user/salford/out/cnn3_10_3.out &
nohup python dqn_her.py -n 10 -r 4 -l 7 -e 100000 -s 10000 > /om/user/salford/out/cnn3_10_4.out &
nohup python dqn_her.py -n 10 -r 6 -l 7 -e 100000 -s 10000 > /om/user/salford/out/cnn3_10_6.out &
nohup python dqn_her.py -n 15 -r 3 -l 7 -e 1000000 -s 10000 > /om/user/salford/out/cnn3_15_3.out &
