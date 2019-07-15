#!/usr/bin/env bash
nohup python dqn_her.py -n 5 -r 2 -e 100 > test1.out &
nohup python dqn_her.py -n 5 -r 2 -e 100 > test2.out &
