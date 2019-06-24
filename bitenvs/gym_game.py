import gym
import numpy as np
from reverse_gym_env import ReverseGymEnv
import sys
sys.path.append('/Users/alfordsimon/Python/Deep_RL_Implementations')
from environments.Bit_Flipping_Environment import Bit_Flipping_Environment

# for testing the correct functionality of the ReverseGymEnv wrapper

def play_game(env):
    print('new game-----------------------------------------------------')
    obs = env.reset()
#    print('action space: ' + str(env.action_space))
    n_actions = env.action_space.n
#    print('state space: ' + str(env.observation_space))

    steps = 0
    while True:
        steps += 1
        obs1 = obs['observation']
        desired = obs['desired_goal']
        print('desired:\t' + str(desired))
        print('obs: \t\t' + str(obs1))
        achieved = obs['achieved_goal']
#        print('achieved:\t' + str(achieved))
        print('pivots\t\t  ' + str(np.array(range(n_actions))))
#        print('optimal? ' + str(optimal_3_agent_action(obs1, desired)))
#        action = optimal_3_agent_action(obs1, desired)
        action = int(input('action: '))
#        print('action: ' + str(action))
        obs, reward, done, _ = env.step(action)
#        print('reward: ' + str(reward))
        if done: 
            return steps

def optimal_3_agent_action(obs, desired):
    i = 0
    while obs[i] == desired[i]:
        i += 1

    # find the later index which has the closest one or zero it needs.
    val = desired[i]
    while obs[i] != val:
        i += 2

    # rotate to make the value go left. index returned is one left of the
    # rotation pivot.
    return i-2
    

def play_game():
    env = ReverseGymEnv(10, 4, 1, 0)
#    env = Bit_Flipping_Environment(9)
    steps = []
    for i in range(1000):
        steps.append(play_game(env))
        print(steps)

    print(steps)



if __name__ == '__main__':
    play_game()
