#!usr/bin/env bash

python -u dqn_her.py --no_save -e 5000 -se 1000 -n 20 -l 1 -t FC --cuda_index 0

