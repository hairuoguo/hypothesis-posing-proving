#!/usr/bin/env python
# coding: utf-8
# 
# Redo of the Pytorch cartpolev0 RL demo to make it operate on the actual
# (position, velocity, reward), etc. tuple instead of looking at the screen.

# now implementing on Hairuo's environment
import sys
sys.path.append('../')

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import bitenvs.uncover_bits_env as ube
import bitenvs.reverse_env as re


agent_name = 'dqn agent'
#env = ube.UncoverBitsEnv(10, 3, 1, 4)
env = re.ReverseEnv(10, 3, 1, 0)
env_name = type(env).__name__ # gives the class name, e.g. 'ReverseEnv'
ep = env.start_ep()

device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

state_space = env.str_len*2+1
action_space = len(ep.actions_list)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        state_space = env.str_len*2+1
        action_space = len(ep.actions_list)
        hidden1 = 128
        hidden2 = 256
        num_hidden = 400
        self.model = torch.nn.Sequential(
                nn.Linear(state_space, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, action_space)
                )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)


BATCH_SIZE = 20
LEARNING_RATE = 0.01
GAMMA = 0.99
# to take random actions
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)
steps_done = 0
episode_durations = []

def select_action(state):
#    state = torch.from_numpy(state).float().unsqueeze(0)

    global steps_done
    sample = random.random()
    eps_threshold = (EPS_END + (EPS_START - EPS_END) 
            * math.exp(-1. * steps_done / EPS_DECAY))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_space)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        # haven't gotten enough experiences in memory yet to replay them
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 500
print_every = 20

for i_episode in range(num_episodes):

    # Initialize the environment and state
    ep = env.start_ep()
    obs1, obs2 = ep.get_obs()
    state = np.concatenate((obs1, np.array([obs2])))
    state = torch.from_numpy(state).float().unsqueeze(0)

    for t in range(500):
        # Select and perform an action
        action = select_action(state)
        obs1, obs2, reward, done = ep.make_action(action)
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = np.concatenate((obs1, np.array([obs2])))
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            break

    episode_durations.append(t+1)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % print_every == 0:
        print('episode ' + str(i_episode) + 'avg duration, past 50: ' +
                str(np.mean(episode_durations[max(0,
                len(episode_durations)-50):])))


def plot_durations(episode_durations):
    # number of episodes for avg
    window = 50

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[6, 6])
    rolling_mean = pd.Series(episode_durations).rolling(window).mean()
    std = pd.Series(episode_durations).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(episode_durations)), rolling_mean -
                     std, rolling_mean+std, color='orange', alpha=0.2)
    ax1.set_title('Episode length rolling 50-avg for ' + env_name + ' with ' 
            + agent_name)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(episode_durations)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)

plot_durations(episode_durations)
