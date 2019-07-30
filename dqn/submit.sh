#!usr/bin/env bash

python -u dqn_her.py --no_save -e 5000 -n 5 -l 1 -t RNN
# nohup python -u dqn_her.py --load --no_save -lm AC_7_3_1 --starting_ep 1000 -nf -e 100 -n 12 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACzomb7_12 &
# nohup python -u dqn_her.py --load --no_save -lm AC_7_3_1 --starting_ep 1000 -nf -e 100 -n 17 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACzomb7_17 &
# nohup python -u dqn_her.py --no_save -lm AC_7_3_1 --starting_ep 1000 -nf -e 100 -n 12 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACnozomb7_12 &
# nohup python -u dqn_her.py --no_save -lm AC_7_3_1 --starting_ep 1000 -nf -e 100 -n 17 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACnozomb7_17 &
# nohup python -u dqn_her.py --load -lm AC_10_3_1_\(7\) --starting_ep 2000 -nf -e 2000 -n 10 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACzomb10_10 &
# nohup python -u dqn_her.py --load -lm AC_10_3_1_\(7\) --starting_ep 2000 -nf -e 2000 -n 11 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACzomb10_11 &
# nohup python -u dqn_her.py --load -lm AC_10_3_1_\(7\) --starting_ep 2000 -nf -e 2000 -n 15 -r 3 -l 1 -t AllConv -f AC_7_3_1 > /om/user/salford/out/ACzomb10_15 &

