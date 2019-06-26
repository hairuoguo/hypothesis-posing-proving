import math
import os
import sys
import bitenvs.reverse_env as re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
#from envs import create_atari_env, create_car_racing_env
from model import ActorCritic
from torch.autograd import Variable
from torch.distributions import Categorical
#from torchvision import datasets, transforms

def sample_subgoals(env, model, path_index, policy_loss, num_subgoals=3):
    cx = Variable(torch.zeros(1, model.lstm_size))
    hx = Variable(torch.zeros(1, model.lstm_size))
    path_state = env.ep.stats.path
    if len(env.ep.stats.path) - path_index > num_subgoals:
        subgoal_state_indices = np.random.choice(range(path_index+1, len(env.ep.stats.path)), size=num_subgoals, replace=False)
        subgoal_states = [env.ep.stats.path[index] for index in subgoal_state_indices]
    else:
        return policy_loss
    subgoals = [state.hidden_state for state in subgoal_states]
    dist_to_subgoals = [(subgoal_index - (path_index)) for subgoal_index in subgoal_state_indices]
    for i in range(len(subgoals)):
        obs, action, reward = env.ep.stats.obs_action_Reward[subgoal_state_indices[i]]
        state_obs = obs[0][:env.str_len]
        subgoal = subgoals[i]
        if np.array_equal(state_obs, subgoal):
            continue
        h_d = re.EpState.get_h_d(state_obs, subgoal)
        obs_input = np.concatenate((state_obs, subgoal, np.aray([h_d])))
        obs_input = torch.from_numpy(obs_input).float().unsqueeze(0)
        probs = policy(obs_input)
        m = Categorical(probs)
        prob = probs[0][action].item()
        action = torch.Tensor([action])
        log_prob = m.log_prob(action)
        reward = reward + her_coeff*(prob)**dist_to_subgoals[i]
        policy_loss = policy_loss - reward
        return policy_loss


def action_bootstrap(env, model, path_index, policy_loss, num_subgoals=1):
    cx = Variable(torch.zeros(1, model.lstm_size))
    hx = Variable(torch.zeros(1, model.lstm_size))
    path_state = env.ep.stats.path[path_index]
    if len(env.ep.stats.path) - path_index > num_subgoals:
        subgoal_state_indices = np.random.choice(range(path_index + 1, len(env.ep.stats.path)), size=num_subgoals, replace=False)
        subgoal_states = [env.ep.stats.path[index] for index in subgoal_state_indices]
    else:
        return policy_loss
    subgoals = [state.hidden_state for state in subgoal_states]
    dist_to_subgoals = [(subgoal_index - (path_index)) for subgoal_index in subgoal_state_indices]
    for i in range(len(subgoals)):
        obs, action, reward = env.ep.stats.obs_action_reward[subgoal_state_indices[i]]
        state_obs = obs[0][:env.str_len]
        subgoal = subgoals[i]
        if np.array_equal(state_obs, subgoal):
            continue
        h_d = re.EpState.get_h_d(state_obs, subgoal)
        obs_input = np.concatenate((state_obs, subgoal, np.array([h_d])))
        value, logit, _ = model((torch.FloatTensor(obs_input), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.multinomial(1).data

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, torch.Tensor([action]).long()).reshape([1])
        policy_loss = policy_loss - loss
        '''
        m = Categorical(prob) 
        sampled_action = m.sample()
        if action == sampled_action:
            advantage = 10 - value
        else:
            advantage = -1 - value
        policy_loss + 0.5*(10-value)**2
        '''
    return policy_loss 

    

def train(args, model, env, optimizer=None):
    torch.manual_seed(args.seed)

    # env = create_atari_env(args.env_name)
    # env = create_car_racing_env()
    env.seed(args.seed)

    num_subgoals = 3
    her_sample = False
    her_coeff = 0.5
    ab = True
    ab_coeff = 1. 


    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    ep = env.start_ep()
    ep_count = 0
    actions_count = 0
    obs1, obs2 = ep.get_obs()
    obs = np.concatenate((obs1, np.array([obs2])))
    done = True

    episode_length = 0
    u = 0
    while u < args.num_updates:
        #print ("update: ", u)
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, model.lstm_size))
            hx = Variable(torch.zeros(1, model.lstm_size))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            actions_count += 1
            value, logit, (hx, cx) = model(
                (Variable(torch.FloatTensor(obs)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, Variable(action))
            obs1, obs2, reward, isEnd = env.ep.make_action(action)
            done = isEnd or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            
            obs = np.concatenate((obs1, np.array([obs2])))
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((torch.FloatTensor(obs), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            if her_sample:
                policy_loss = sample_subgoals(env, model, i, policy_loss)
            if ab:
                policy_loss = action_bootstrap(env, model, i, policy_loss)
            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        optimizer.step()
        ep_count += 1
        episode_length = 0
        ep = env.start_ep()
        obs1, obs2 = ep.get_obs()
        u += 1
    return actions_count/ep_count
