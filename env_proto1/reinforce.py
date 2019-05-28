import uncover_bits_env as ube
import argparse
import identity_env as ie
import simple_env as se
import numpy as np
from scipy import stats
from itertools import count
import random

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.5)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = ube.UncoverBitsEnv(10, 3, 1, 4)
ep = env.start_ep()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(env.str_len+1, 128)
        self.affine2 = nn.Linear(128, len(ep.actions_list))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(obs_input):
    obs_input = np.concatenate((obs_input[0], np.array([obs_input[1]])))
    obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
    probs = policy(obs_input)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def run_on_env(num_eps=100000, max_num_actions=500, test=False, plot=False):
    avg_num_actions = 0
    list_t = []
    for i_episode in range(num_eps):
        ep = env.start_ep()
        obs1, obs2 = ep.get_obs()
        for t in range(1, max_num_actions+1):
            action = select_action((obs1, obs2))
            obs1, obs2, reward, is_end = ep.make_action(action)
            policy.rewards.append(reward)
            if is_end:
                break
        if not test:
            finish_episode()
        avg_num_actions += t
        list_t.append(t)
        if i_episode % 100 == 0:
            print("Episode: %s, avg num actions: %s" % (i_episode, avg_num_actions/100))
            avg_num_actions = 0
    if plot:
        fig, ax = plt.subplots()
        ax.plot([i for i in range(num_eps)], list_t)
        plt.show()

'''
def test_on_env():
    num_eps = 10000
    num_actions = 100
    avg_num_actions = 0
    list_t = []
    successes = 0
    for i_episode in range(num_eps):
        ep = env.start_ep()
        obs = ep.get_obs()
        for t in range(1, num_actions + 1):
            action = select_action(obs)
            obs, reward, is_end = ep.make_action(action)
            if is_end:
                successes += 1
                break
        avg_num_actions += t
        list_t.append(t)
        if i_episode % 100 == 0:
            state = torch.from_numpy(obs_input).float().unsqueeze(0)
            probs = policy(state)
            avg_num_actions = 0
    fig, ax = plt.subplots()
    ax.plot([i for i in range(num_eps)], list_t)
    plt.show()
'''
'''
def main():
    running_reward = 10
    for i_episode in count(10000):
        ep = env.start_ep()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
'''

if __name__ == '__main__':
    run_on_env() 
    torch.manual_seed(args.seed)
