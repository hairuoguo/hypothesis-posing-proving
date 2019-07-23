#!usr/bin/env bash
# nohup python -u dqn_her.py -e 100000 -n 10 -r 4 -l 3 -t CNN -f CNN_10_4_L3 -nf > /om/user/salford/out/CNN_10_4_L3.out &
# nohup python -u dqn_her.py -e 100000 -n 10 -r 4 -l 3 -t ResNet --num_filters 100 --num_blocks 1 -f ResNet100f_10_4_L3 -nf > /om/user/salford/out/Resnet100f_10_4_L3.out &
# nohup python -u dqn_her.py -e 100000 -n 10 -r 4 -l 3 -t ResNet --num_filters 10 --num_blocks 1 -f ResNet10f_10_4_L3 -nf > /om/user/salford/out/Resnet10f_10_4_L3.out &
# nohup python -u dqn_her.py -e 100000 -n 10 -r 4 -l 3 -t FC -f FC_10_4_L3 -nf > /om/user/salford/out/FC_10_4_L3 &
nohup python -u dqn_her.py -e 100000 -n 10 -r 4 -l 3 -t AllConv -f AllConv0_10_4_L3 -nf > /om/user/salford/out/AllConv0_10_4_L3.out &
