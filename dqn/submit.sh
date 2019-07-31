#!usr/bin/env bash

#nohup python -u dqn_her.py --num_filters 200 --num_blocks 3 -e 25000 --save_every 1000 -n 10 -r 4 -l 3 -t AllConv --cuda_index 0 > /om/user/salford/out/all_conv_baseline3.out &
#python -u dqn_her3.py --no_save -e 5000 -se 1000 -n 10 -l 1 -t RNN --cuda_index 0 --binary_env --DQN
python -u dqn_her3.py --no_save -e 5000 -se 1000 -n 20 -l 1 -t FC --cuda_index 0 --binary_env --DQN


