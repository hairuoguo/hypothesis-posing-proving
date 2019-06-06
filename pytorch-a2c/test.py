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
import time
from collections import deque


def test(args, model, env):
    torch.manual_seed(args.seed)

    # env = create_atari_env(args.env_name)
    # env = create_car_racing_env()
    env.seed(args.seed)


    model.eval()
    
    ep = env.start_ep()
    obs1, obs2 = ep.get_obs()
    obs = np.concatenate((obs1, np.array([obs2])))
    done = True

    reward_sum = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        #env.render()
        episode_length += 1
        # Sync with the shared model
        if done:
            # model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, model.lstm_size), volatile=True)
            hx = Variable(torch.zeros(1, model.lstm_size), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model((
            Variable(torch.FloatTensor(obs), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()
        obs1, obs2, reward, isEnd = env.ep.make_action(action[0])
        done = isEnd or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            ep = env.start_ep()
            obs1, obs2 = ep.get_obs()
            return
            # time.sleep(60)

        obs = np.concatenate((obs1, np.array([obs2])))
