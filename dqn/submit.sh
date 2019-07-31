#!usr/bin/env bash

nohup python -u dqn_her.py -e 5000 -n 10 -nf -l 1 -t RNN -se 1000 --cuda_index 0 -f lstm_test > /om/user/salford/out/lstm_test.out &

