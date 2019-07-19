#!usr/bin/env bash
nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres_10_6 &
nohup python -u dqn_her.py -n 15 -r 6 -e 500000 --cuda_index 4 --num_blocks 1 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres_15_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres3_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres3_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres3_10_6 &
nohup python -u dqn_her.py -n 15 -r 6 -e 500000 --cuda_index 4 --num_blocks 3 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres3_15_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --num_blocks 10 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres0_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --num_blocks 10 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres0_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --num_blocks 10 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres0_10_6 &
nohup python -u dqn_her.py -n 15 -r 6 -e 500000 --cuda_index 4 --num_blocks 10 --num_filters 256 --net_type ResNet > /om/user/salford/out/comppres0_15_6 &

nohup python -u dqn_her.py -n 10 -r 3 -e 100000 --cuda_index 1 --net_type FC > /om/user/salford/out/comppfc_10_3 &
nohup python -u dqn_her.py -n 15 -r 3 -e 500000 --cuda_index 2 --net_type FC > /om/user/salford/out/comppfc_15_3 &
nohup python -u dqn_her.py -n 10 -r 6 -e 500000 --cuda_index 3 --net_type FC > /om/user/salford/out/comppfc_10_6 &
nohup python -u dqn_her.py -n 15 -r 6 -e 500000 --cuda_index 4 --net_type FC > /om/user/salford/out/comppfc_15_6 &
