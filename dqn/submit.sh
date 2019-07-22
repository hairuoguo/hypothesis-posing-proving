#!usr/bin/env bash
nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res_10_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res3_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res3_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res3_10_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 6 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res0_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 6 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res0_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 6 --num_filters 256 --net_type ResNet > /om/user/salford/out/comp3res0_10_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --net_type FC > /om/user/salford/out/comp3fc_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --net_type FC > /om/user/salford/out/comp3fc_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --net_type FC > /om/user/salford/out/comp3fc_10_6 &

