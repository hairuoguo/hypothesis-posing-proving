#!usr/bin/env bash

python -u dqn_her.py -e 5000 -se 500 -n 10 -l 2 -t FC --cuda_index 7
--env binary -f rnn_testing


