#!usr/bin/env bash

nohup python -u random_search.py --cuda_index 7 --id 0 --net_type RNN > /om/user/salford/out/random_out1.out &
nohup python -u random_search.py --cuda_index 7 --id 1 --net_type RNN > /om/user/salford/out/random_out2.out &
nohup python -u random_search.py --cuda_index 7 --id 2 --net_type RNN > /om/user/salford/out/random_out3.out &
nohup python -u random_search.py --cuda_index 7 --id 3 --net_type RNN > /om/user/salford/out/random_out4.out &
nohup python -u random_search.py --cuda_index 7 --id 4 --net_type RNN > /om/user/salford/out/random_out5.out &


