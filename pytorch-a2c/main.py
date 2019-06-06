from __future__ import print_function

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
#from envs import create_atari_env, create_car_racing_env
from envs import *
from model import ActorCritic
from train import train
from test import test

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-steps', type=int, default=100, metavar='NS',
                    help='number of forward steps in A3C (default: 500)')
parser.add_argument('--num-updates', type=int, default=50, metavar='NU',
                    help='number of updates between tests (default: 100)')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='PongDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    str_len = 10
    #env = create_uncover_bits_env(str_len, 3, 1, 4)
    env = create_noocc_env(10, 3, 1)
    env.start_ep()

    # env = create_atari_env(args.env_name)
    #env = create_car_racing_env()
    action_keys = range(len(env.ep.actions_list)) 
    model = ActorCritic(str_len*2+1, action_keys)

    itr = 0
    while itr < 10000/args.num_updates:
        avg_num_actions = train(args, model, env)
        print("Num updates: %s, avg num actions: %s" % ((itr+1)*args.num_updates, avg_num_actions))
        #test(args, model, env)
        itr += 1
