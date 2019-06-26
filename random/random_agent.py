    #!/usr/bin/env python
# coding: utf-8
# Random agent for bit environment
import sys
sys.path.append('../')

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import pandas as pd

import bitenvs.uncover_bits_env as ube
import bitenvs.reverse_env as re


agent_name = 'random agent'
#env = ube.UncoverBitsEnv(10, 3, 1, 4)
env = re.ReverseEnv(10, 3, 1, 0)
env_name = type(env).__name__ # gives the class name, e.g. 'ReverseEnv'
ep = env.start_ep()

state_space = env.str_len*2+1
action_space = len(ep.actions_list)

steps_done = 0

def select_action(state):
    return random.randint(0, action_space-1)

episode_durations = []

num_episodes = 500
print_every = 10

for i_episode in range(num_episodes):
    # Initialize the environment and state
    ep = env.start_ep()
    obs1, obs2 = ep.get_obs()
    state = obs1, obs2

    for t in range(500):
        # Select and perform an action
        action = select_action(state)
        obs1, obs2, reward, done = ep.make_action(action)

        if done:
            break

    episode_durations.append(t+1)

    if i_episode % print_every == 0:
        print('episode ' + str(i_episode) + 'avg duration, past 50: ' +
                str(np.mean(episode_durations[max(0,
                len(episode_durations)-50):])))

def plot_durations(episode_durations):
    # number of episodes for rolling average
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
plt.show()
