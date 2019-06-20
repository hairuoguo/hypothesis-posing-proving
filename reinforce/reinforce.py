import sys
sys.path.append('../')

import bitenvs.uncover_bits_env as ube
import bitenvs.reverse_env as re
import argparse
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

bitstring_len = 10
#env = ube.UncoverBitsEnv(bitstring_len, 3, 1, 6)
env = re.ReverseEnv(10, 3, 1, 0)
ep = env.start_ep()
num_subgoals = 1
her_sample = True


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(env.str_len*2+1, 128)
        
        #self.affine2 = nn.Linear(128, 256)
        #self.affine3 = nn.Linear(256, len(ep.actions_list))
        
        self.affine2 = nn.Linear(128, len(ep.actions_list))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        #x1 = F.relu(self.affine1(x))
        #x2 = F.relu(self.affine2(x1))
        #action_scores = self.affine3(x2)
        
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        
        return F.softmax(action_scores, dim=1)

lr=1e-2
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()

def select_action(obs_input):
    obs_input = np.concatenate((obs_input[0], np.array([obs_input[1]])))
    obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
    probs = policy(obs_input)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def sample_subgoals(path_index, policy_loss):
    path_state = env.ep.stats.path[path_index]
    if len(env.ep.stats.path) - path_index > num_subgoals:
        subgoal_state_indices = np.random.choice(range(path_index+1, len(env.ep.stats.path)), size=num_subgoals, replace=False)
        subgoal_states = [env.ep.stats.path[index] for index in subgoal_state_indices]
    else:
        return
    subgoals = [state.hidden_state for state in subgoal_states]
    dist_to_subgoals = [(subgoal_index - (path_index + 1)) for subgoal_index in subgoal_state_indices]
    for i in range(len(subgoals)):
        obs, action, reward = env.ep.stats.obs_action_reward[i]
        h_d = re.EpState.get_h_d(obs[0][:bitstring_len], subgoals[i])
        obs_input = np.concatenate((obs[0][:bitstring_len], subgoals[i], np.array([h_d])))
        obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
        probs = policy(obs_input)
        m = Categorical(probs)
        action = torch.Tensor([action])
        log_prob = m.log_prob(action)
        reward = reward + args.gamma**dist_to_subgoals[i]
        policy_loss.append(-log_prob * reward)


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    policy_rewards = policy.rewards[::-1]
    for r in policy_rewards:
        R = r + args.gamma * R
        rewards.insert(0, R)
        sample_subgoals(policy_rewards.index(r), policy_loss)
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


def run_on_env(num_eps=10000, max_num_actions=500, test=False, plot=False):
    print("lr: %s, num_eps: %s, max_num_actions: %s" % (lr, num_eps, max_num_actions))
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
