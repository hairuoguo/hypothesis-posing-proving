import gym
import numpy as np
from reverse_gym_env import ReverseGymEnv

# for testing the correct functionality of the ReverseGymEnv wrapper

def play_game(env):
    obs = env.reset()
    print('action space: ' + str(env.action_space))
    n_actions = env.action_space.n
    print('state space: ' + str(env.observation_space))

    while True:
        obs1 = obs['observation']
        print('obs1: \t\t' + str(obs1))
        desired = obs['desired_goal']
        print('desired:\t' + str(desired))
        achieved = obs['achieved_goal']
        print('achieved:\t' + str(achieved))
        print('pivots\t\t  ' + str(np.array(range(n_actions))))
        action = int(input('choose action:'))
        action = action % n_actions
        print(action)
        obs, reward, done, _ = env.step(action)
        if done: 
            break

if __name__ == '__main__':
    env = ReverseGymEnv(10, 3, 1, 0)
    play_game(env)
