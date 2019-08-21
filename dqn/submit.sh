#!usr/bin/env bash
nohup python -u dqn_her.py -e 100000 -n 7 -r 3 -l 10 -t FC -se 5000 -f 'fc_degrade'> /om/user/salford/out/fc_degrade1.out &
nohup python -u dqn_her.py -e 100000 -n 7 -r 3 -l 10 -t FC -se 5000 -f 'fc_degrade1'> /om/user/salford/out/fc_degrade2.out &
nohup python -u dqn_her.py -e 100000 -n 7 -r 3 -l 10 -t FC -se 5000 -f 'fc_degrade2'> /om/user/salford/out/fc_degrade3.out &
nohup python -u dqn_her.py -e 100000 -n 7 -r 3 -l 10 -t FC -se 5000 -f 'fc_degrade3'> /om/user/salford/out/fc_degrade4.out &
