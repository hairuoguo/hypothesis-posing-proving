import copy
import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

class BinaryEnv(gym.Env):
    environment_name = "Binary Environment"

    def __init__(self, environment_dimension=2, num_to_flip=1,
            deterministic=False):

        self.action_space = spaces.Discrete(environment_dimension)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(0, 1, shape=(environment_dimension,), dtype='float32'),
            achieved_goal=spaces.Box(0, 1, shape=(environment_dimension,), dtype='float32'),
            observation=spaces.Box(0, 1, shape=(environment_dimension,), dtype='float32'),
        ))

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 50
        self.max_episode_steps = environment_dimension
        self.id = "Bit Flipping"
        self.environment_dimension = environment_dimension
        self.num_to_flip = num_to_flip
        self.reward_for_achieving_goal = self.environment_dimension
        self.step_reward_for_not_achieving_goal = -1
        self.deterministic = deterministic

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.is_solved = False

        if not self.deterministic:
            self.desired_goal = self.make_desired_goal()
            self.state = self.make_state()
        else:
            self.desired_goal = [0 for _ in range(self.environment_dimension)]
            self.state = [1 for _ in range(self.environment_dimension)]
        self.state.extend(self.desired_goal)
        self.achieved_goal = self.state[:self.environment_dimension]
        self.step_count = 0
        return {"observation": np.array(self.state[:self.environment_dimension]), "desired_goal": np.array(self.desired_goal),
                "achieved_goal": np.array(self.achieved_goal)}

    def randomly_pick_state_or_goal(self):
        return [random.randint(0, 1) for _ in range(self.environment_dimension)]

    def make_desired_goal(self):
        return [0]*self.environment_dimension

    def make_state(self):
        a = [0]*self.environment_dimension
        for i in random.sample(range(0, self.environment_dimension),
                self.num_to_flip):
            a[i] = 1
        return a

    def step(self, action):
        """Conducts the discrete action chosen and updated next_state, reward and done"""
        if type(action) is np.ndarray:
            action = action[0]
        assert action <= self.environment_dimension + 1, "You picked an invalid action"
        self.step_count += 1
        self.next_state = copy.copy(self.state)
        self.next_state[action] = 0
        if self.goal_achieved(self.next_state):
            self.reward = self.reward_for_achieving_goal
            self.done = True
            self.is_solved = True
        else:
            self.reward = self.step_reward_for_not_achieving_goal
            if self.step_count >= self.environment_dimension + 2:
                self.done = True
            else:
                self.done = False
        self.achieved_goal = self.next_state[:self.environment_dimension]
        self.state = self.next_state

        return {"observation": np.array(self.next_state[:self.environment_dimension]),
                "desired_goal": np.array(self.desired_goal), "achieved_goal": np.array(self.achieved_goal)}, self.reward, self.done, {}

    def goal_achieved(self, next_state):
        return next_state[:self.environment_dimension] == next_state[-self.environment_dimension:]

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward
