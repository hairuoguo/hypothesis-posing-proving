import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import sys
import random as rand
from bitenvs.reverse_env import ReverseEnv
 
# a wrapper over ReverseEnv that extends gym.Env to allow running DQN_HER 
# easily on it. Based off
# Deep_RL_Implementations/environments/Bit_Flipping_Environment.py

class ReverseGymEnv(gym.Env):
    environment_name = "Reverse Bit Game"

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured,
            hypothesis_enabled=False, path_len_mean=5, path_len_std=0.5,
            print_results=True):

        self.env = ReverseEnv(str_len, reverse_len, reverse_offset,
                num_obscured, hypothesis_enabled=hypothesis_enabled,
                path_len_mean=path_len_mean, path_len_std=path_len_std,
                print_results=print_results)
        self.str_len = str_len
        self.action_space = spaces.Discrete(len(self.env.actions_list))
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(0, 1, shape=(self.str_len,), dtype='float32'),
            achieved_goal=spaces.Box(0, 1, shape=(self.str_len,), dtype='float32'),
            observation=spaces.Box(0, 1, shape=(self.str_len,), dtype='float32'),
        ))

        self.seed()
        # avg score required to trigger a 'win'. I
        # this only affects the learning rate which Base Agent.py adjusts if
        # you're getting closer to the reward threshold.
        self.reward_threshold = str_len
        self.trials = 50 # num of trials to avg over
        self.id = f'ReverseEnv: ({str_len}, {reverse_len}, {reverse_offset}, {num_obscured})'
        self.reward_for_achieving_goal = self.env.reward_for_achieving_goal
        self.step_reward_for_not_achieving_goal = self.env.step_reward_for_not_achieving_goal
        self.is_solved = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Should return a state_dict with keys 'observation', 'desired_goal', 
        and 'achieved_goal'
        """
        self.ep = self.env.start_ep()
#        self.max_episode_steps = self.ep.path_len + 1
        self.max_episode_steps = self.str_len
#        self.max_episode_steps = self.ep.path_len*2 + 1
        self.step_count = 0
        # obs1 is concatenation of current and target state. obs2 is l1
        # distance)
        obs1, l1 = self.ep.get_obs()
        self.state = obs1
        print('start stat: {}'.format(obs1[:self.str_len]))
        print('start goal: {}'.format(obs1[self.str_len:]))
        self.desired_goal = obs1[self.str_len:]
        self.achieved_goal = obs1[:self.str_len]
        self.is_solved = False


        return {"observation": np.array(obs1[:self.str_len]), "desired_goal":
                np.array(self.desired_goal),
                "achieved_goal": np.array(self.achieved_goal)}

    def step(self, action):
        obs1, l1, reward, isEnd = self.ep.make_action(action)
        self.step_count += 1

        self.next_state =  obs1
        self.done = isEnd or self.step_count >= self.max_episode_steps
        self.achieved_goal = obs1[:self.str_len]
        self.state = self.next_state
        self.is_solved = isEnd

        print('after move: {}'.format(obs1[:self.str_len]))
        print('goal      : {}'.format(self.desired_goal))
        return ({"observation": np.array(obs1[:self.str_len]),
                "desired_goal": np.array(self.desired_goal),
                "achieved_goal": np.array(self.achieved_goal)}, reward,
            self.done, {})

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward
