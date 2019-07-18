#!usr/bin/env bash
#nohup python dqn_her.py -n 10 -r 6 -e 50000 --save_every 5000 --net_type ResNet > /om/user/salford/out/res_10_6.out &
#nohup python dqn_her.py -n 15 -r 3 -e 50000 --save_every 5000 --net_type ResNet > /om/user/salford/out/res_15_3.out &
#nohup python dqn_her.py -n 10 -r 6 -e 50000 --save_every 5000 --net_type FC > /om/user/salford/out/fc_10_6.out &
#nohup python dqn_her.py -e 5000 --no_save --cuda_index 4 > /om/user/salford/out/test2.out &
python dqn_her.py -e 50 --no_save --cuda_index 4 > /om/user/salford/out/test33.out
python dqn_her.py -e 50 --no_save --cuda_index 4
