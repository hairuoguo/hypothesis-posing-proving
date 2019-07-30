#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../')

from bitenvs import reverse_env as rev
from bitenvs import uncover_bits_env as ube

import numpy as np
import getopt
import os
import math
import time
import argparse

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from dnc.dnc import DNC
from dnc.sdnc import SDNC
from dnc.sam import SAM
from dnc.util import *
#from visdom import Visdom


parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-input_size', type=int, default=6, help='dimension of input feature')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nlayer', type=int, default=2, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=20, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

parser.add_argument('-sequence_max_length', type=int, default=1000, metavar='N', help='sequence_max_length')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=1000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')
parser.add_argument('-gamma', type=float, default=0.25, help='gamma value for reward decay')

args = parser.parse_args()
print(args)

dnc_rewards = []
dnc_saved_log_probs = []

bit_str_len = 4
reverse_len = 2
path_len_mean = 1
path_len_std = 0

env = rev.ReverseEnv(bit_str_len, reverse_len, 1, 0,
        path_len_mean=path_len_mean, path_len_std=path_len_std)
ep = env.start_ep()
num_subgoals = 3
#her_sample = False
her_coeff = 1.
ab = False
rnn = DNC(
    input_size=bit_str_len*2+1,
    hidden_size=len(env.ep.actions_list),
    rnn_type=args.rnn_type,
    num_layers=args.nlayer,
    num_hidden_layers=args.nhlayer,
    dropout=args.dropout,
    nr_cells=args.mem_slot,
    cell_size=args.mem_size,
    read_heads=args.read_heads,
    gpu_id=args.cuda,
    debug=args.visdom,
    batch_first=True,
    independent_linears=True
)

if args.cuda != -1:
    rnn = rnn.cuda(args.cuda)

    print(rnn)


if args.optim == 'adam':
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
elif args.optim == 'adamax':
    optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)  # 0.0001
elif args.optim == 'sgd':
    optimizer = optim.SGD(rnn.parameters(), lr=args.lr)  # 0.01
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
elif args.optim == 'adadelta':
    optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)
# assert viz.check_connection()

if args.cuda != -1:
    print('Using CUDA.')
    T.manual_seed(1111)
else:
    print('Using CPU.')

def finish_episode():
    #print('here')
    R = 0
    policy_loss = []
    rewards = []
    T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
    rev_dnc_rewards = dnc_rewards[::-1]
    for i in range(len(rev_dnc_rewards)):
        r = rev_dnc_rewards[i]
        R = r + args.gamma * R
        rewards.insert(0, R)
        if her_sample:
            sample_subgoals(i, policy_loss)
        if ab:
            action_bootstrap(i, policy_loss)
    rewards = torch.tensor(rewards)
    for log_prob, reward in zip(dnc_saved_log_probs, rewards):
        loss = -log_prob*reward
        #if not T.eq(loss, T.zeros([1, 1])):
        policy_loss.append(loss)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    #print(policy_loss)
    policy_loss.backward(retain_graph=True)
    optimizer.step()
    del dnc_rewards[:]
    del dnc_saved_log_probs[:]

def select_action(probs):
    #print(probs)
    m = Categorical(probs)
    action = m.sample()
    dnc_saved_log_probs.append(m.log_prob(action))
    return action.item()


def sample_subgoals(path_index, policy_loss):
    path_state = env.ep.stats.path[path_index]
    if len(env.ep.stats.path) - path_index > num_subgoals:
        subgoal_state_indices = np.random.choice(range(path_index+1, len(env.ep.stats.path)), size=num_subgoals, replace=False)
        subgoal_states = [env.ep.stats.path[index] for index in subgoal_state_indices]
    else:
        return
    subgoals = [state.hidden_state for state in subgoal_states]
    dist_to_subgoals = [(subgoal_index - (path_index)) for subgoal_index in subgoal_state_indices]
    for i in range(len(subgoals)):
        obs, action, reward = env.ep.stats.obs_action_reward[subgoal_state_indices[i]]
        state_obs = obs[0][:bit_str_len]
        subgoal = subgoals[i]
        if np.array_equal(state_obs, subgoal):
            continue
        h_d = rev.EpState.get_h_d(state_obs, subgoal)
        obs_input = np.concatenate((state_obs, subgoal, np.array([h_d]))).reshape((1, 1, -1))
        #obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
        probs, (chx, m, rv) = rnn(Variable(torch.FloatTensor(obs_input)), (None, mhx, None), reset_experience=True, pass_through_memory=True)
        m = Categorical(probs)
        action = torch.Tensor([action])
        log_prob = m.log_prob(action)
        reward = reward + her_coeff*args.gamma**dist_to_subgoals[i]
        policy_loss.append(-log_prob*reward)         
    

def action_bootstrap(path_index, policy_loss):
    path_state = env.ep.stats.path[path_index]
    if len(env.ep.stats.path) - path_index > num_subgoals:
        #subgoal_state_indices = [path_index + 1]
        subgoal_state_indices = np.random.choice(range(path_index+1, len(env.ep.stats.path)), size=num_subgoals, replace=False)
        subgoal_states = [env.ep.stats.path[index] for index in subgoal_state_indices]
    else:
        return
    subgoals = [state.hidden_state for state in subgoal_states]
    dist_to_subgoals = [(subgoal_index - (path_index)) for subgoal_index in subgoal_state_indices]
    for i in range(len(subgoals)):
        obs, action, reward = env.ep.stats.obs_action_reward[subgoal_state_indices[i]]
        state_obs = obs[0][:bit_str_len]
        subgoal = subgoals[i]
        if np.array_equal(state_obs, subgoal):
            continue
        h_d = rev.EpState.get_h_d(state_obs, subgoal)
        obs_input = np.concatenate((state_obs, subgoal, np.array([h_d]))).reshape((1, 1, -1))
        #obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
        probs, (chx, m, rv) = rnn(Variable(torch.FloatTensor(obs_input)), (None, mhx, None), reset_experience=True, pass_through_memory=False)
        m = Categorical(probs)
        '''
        criterion = T.nn.CrossEntropyLoss()
        loss = criterion(probs.reshape((1, len(env.ep.actions_list))), T.Tensor([action]).long()).reshape((1, 1))
        policy_loss.append(-1*loss)
        '''       
        
        sampled_action = m.sample()
        if action == sampled_action:
            action = torch.Tensor([action])
            log_prob = m.log_prob(action)
            reward = dist_to_subgoals[i]*1
            policy_loss.append(-log_prob * reward)
            #print(-log_prob * reward)
        else:
            log_prob = m.log_prob(sampled_action)
            reward = -10*dist_to_subgoals[i]
            policy_loss.append(-log_prob*reward)

if __name__ == '__main__':


    from_checkpoint = None

#    viz = Visdom()


    isEnd = False
    
    i = 0

    num_eps = 100000

    max_steps = 5
    
    ep = env.start_ep()

    obs1, obs2 = ep.get_obs()

    obs = np.concatenate((obs1, np.array([obs2]))).reshape((1, 1, -1))     
    
    num_actions = 0

    while i < num_eps:

        i+=1

        (chx, mhx, rv) = (None, None, None)

        for step in range(max_steps):
            num_actions += 1
        
            optimizer.zero_grad()
            if rnn.debug:
                output, (chx, mhx, rv), v = rnn(Variable(torch.FloatTensor(obs)), (None, mhx, None), reset_experience=False, pass_through_memory=True)
            else:
                output, (chx, mhx, rv) = rnn(Variable(torch.FloatTensor(obs)), (None, mhx, None), reset_experience=False, pass_through_memory=True)

            if i % 100 == 0: print(output)

            action = select_action(output)
            obs1, obs2, reward, isEnd = ep.make_action(action)
            dnc_rewards.append(reward) 


            # detach memory from graph
            mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }
            
            if isEnd or step == (max_steps - 1):
                finish_episode()
                ep = env.start_ep()
                obs = np.concatenate((obs1, np.array([obs2]))).reshape((1, 1, -1))
                break

        take_checkpoint = (i != 0) and (i % args.iterations == 0)

        '''
        if take_checkpoint:
            print("\nSaving Checkpoint ... "),
            check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
            cur_weights = rnn.state_dict()
            T.save(cur_weights, check_ptr)
            print("Done!\n")
        '''

        if i % 100 == 0:
            print("Episode: %s, avg num actions: %s" % (i, num_actions/100))
            num_actions = 0 
        


     

    
