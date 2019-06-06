import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
#from envs import create_atari_env, create_car_racing_env
from model import ActorCritic
from torch.autograd import Variable
#from torchvision import datasets, transforms

def train(args, model, env, optimizer=None):
    torch.manual_seed(args.seed)

    # env = create_atari_env(args.env_name)
    # env = create_car_racing_env()
    env.seed(args.seed)


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
            
            if done:
                ep_count += 1
                episode_length = 0
                ep = env.start_ep()
                obs1, obs2 = ep.get_obs()
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

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        optimizer.step()
        u += 1
    return actions_count/ep_count
