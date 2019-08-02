#!usr/bin/env bash

nohup python -u random_search.py --cuda_index 7 --id 1 --net_type RNN > /om/user/salford/out/rand_search_1.out &
nohup python -u random_search.py --cuda_index 7 --id 2 --net_type RNN > /om/user/salford/out/rand_search_2.out &
nohup python -u random_search.py --cuda_index 7 --id 3 --net_type RNN > /om/user/salford/out/rand_search_3.out &
nohup python -u random_search.py --cuda_index 7 --id 4 --net_type CNN > /om/user/salford/out/rand_search_4.out &
nohup python -u random_search.py --cuda_index 7 --id 5 --net_type CNN > /om/user/salford/out/rand_search_5.out &
                                                                                                
