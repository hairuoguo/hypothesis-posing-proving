#!usr/bin/env bash

nohup python -u dqn_her.py -nf -e 2000 --no_save -se 2000 -n 10 -r 3 -l 3 -t AllConv -f ACzomb_7_10 > /om/user/salford/out/ACzomb_7_10
#nohup python -u dqn_her.py -nf -e 20000 -se 2000 -n 10 -r 3 -l 1 -t AllConv -f AC_10_3_1 > /om/user/salford/out/AC_10_3_1 &
#nohup python -u dqn_her.py -nf -e 20000 -se 2000 -n 10 -r 3 -l 3 -t AllConv -f AC_10_3_3 > /om/user/salford/out/AC_10_3_3 &
#nohup python -u dqn_her.py -nf -e 20000 -se 2000 -n 15 -r 3 -l 1 -t AllConv -f AC_15_3_1 > /om/user/salford/out/AC_15_3_1 &
#nohup python -u dqn_her.py -nf -e 50000 -se 2000 -n 15 -r 3 -l 3 -t AllConv -f AC_15_3_3 > /om/user/salford/out/AC_15_3_3 &

