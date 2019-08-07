import sys
sys.path.append('../')
import os
from deep_rl.agents.DQN_agents.DQN_HER import DQN_HER
from deep_rl.agents.DQN_agents.DQN import DQN
from deep_rl.agents.Trainer import Trainer
from deep_rl.utilities.data_structures.Config import Config
from deep_rl.environments.Bit_Flipping_Environment import Bit_Flipping_Environment
from bitenvs.reverse_gym_env import ReverseGymEnv
from bitenvs.binary_gym_env import BinaryEnv
import deep_rl.utilities.file_numberer as file_numberer
import argparse
import torch
import deep_rl.utilities.info_maker as info_maker
from pathlib import Path
from numpy.random import choice
import numpy as np
import math


def run_random_search_forever(cuda_index, id, net_type):
    while True:
        make_and_run_random_instance(cuda_index, id, net_type)

def loguniform(low=-4, high=-1, size=None):
        return math.pow(10, np.random.uniform(low, high, size))

def make_and_run_random_instance(cuda_index, id, net_type):
    str_len = choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
    path_len = choice(range(1, str_len+1))
    max_episode_steps = choice(range(path_len, 2*path_len+2))
    learning_rate = loguniform(low=-4, high=-1)
    batch_size = choice([4, 16, 32, 64, 128, 128])
    learning_iterations= choice([1, 1, 1, 2, 4, 8])
    env = choice(['binary','reverse'], p=[0.5, 0.5])

    if env == 'reverse':
        str_len = choice([7, 8, 9, 10])
        path_len = choice(range(1, str_len+1))
        max_episode_steps = choice(range(path_len, 2*path_len+2))

    run_random_instance(net_type, str_len, path_len, max_episode_steps,
            learning_rate, batch_size, learning_iterations, env, cuda_index, id)



def run_random_instance(net_type, str_len, path_len, max_episode_steps,
        learning_rate, batch_size, learning_iterations, env, cuda_index, id):
    info = ('random rnn search: \nnet_type = ' + str(net_type) 
            + ', str_len = ' + str(str_len)
            + ', path_len = ' + str(path_len)
            + ', max ep steps = ' + str(max_episode_steps)
            + ', learning_rate = ' + str(learning_rate)
            + ', batch_size = ' + str(batch_size)
            + ', learning steps per iteration = ' + str(learning_iterations)
            + ', cuda_index = ' + str(cuda_index)
            + ', env = ' + str(env)
            + ', id = ' + str(id))

    print('running new search\ninfo=\n' + info)
    run_dqn_her(num_eps=50000, 
            str_len=str_len,
            path_len=path_len,
            save_every=5000,
            file_name= net_type + '_rand2_c'+ str(cuda_index) + '_i' + str(id),
            info=info,
            cuda_index=cuda_index,
            net_type=net_type,
            env=env,
            max_episode_steps=max_episode_steps,
            learning_rate = learning_rate,
            batch_size = batch_size,
            learning_iterations=learning_iterations)


def run_dqn_her(num_eps=50000,
        str_len=10,
        reverse_len=3,
        path_len=3,
        save_every=5000, 
        file_name='',
        info='', 
        cuda_index=1,
        num_blocks=1,
        num_filters=10,
        net_type='RNN',
        load_model='',
        starting_ep=0,
        no_save=False,
        no_flush=False,
        load=False,
        env='binary',
        max_episode_steps=1,
        learning_rate=0.01,
        batch_size=128,
        learning_iterations=1):

    config = Config()

    str_len = str_len
    reverse_len = reverse_len
    config.reverse_len = reverse_len
    reverse_offset = 1
    num_obscured = 0
    path_len_mean = path_len
    path_len_std = 0
    max_episode_steps= max_episode_steps

    if env == 'reverse':
        env = ReverseGymEnv(str_len, reverse_len, reverse_offset, num_obscured,
                max_episode_steps=max_episode_steps,
                hypothesis_enabled=False, path_len_mean=path_len_mean,
                path_len_std=path_len_std, print_results=False)

    elif env == 'binary':
        env = BinaryEnv(str_len, path_len_mean,
                max_episode_steps=max_episode_steps) 

    elif env == 'uncover':
        env = UncoverGymEnv(str_len, reverse_len, reverse_offset, num_obscured)

    config.save_every_n_episodes = save_every
    config.save_results = not no_save

    data_dir = '/om/user/salford/data'
    model_dir = '/om/user/salford/models'
    plot_dir = '/om/user/salford/plots'
    info_dir = '/om/user/salford/info'

    if not no_save:
        if len(file_name) > 0:
            model_name = file_name
        else:
            model_name = str.format('{}_{}_{}_L{}', net_type,
                    str_len, reverse_len, path_len_mean)

        (config.file_to_save_data_results,
         config.file_to_save_model,
         config.file_to_save_results_graph,
         config.file_to_save_session_info) = file_numberer.get_unused_filepaths(
                 model_name, data_dir, model_dir, plot_dir, info_dir)

        # Immediately mark file as used so that other programs don't think it's untaken
        # in case we run lots of jobs at the same time
        Path(config.file_to_save_session_info).touch()

        config.info = info

    config.environment = env
    config.no_random = False # if True, disables random actions but still trains
    config.num_episodes_to_run = num_eps
    config.cuda_index = cuda_index
    if torch.cuda.is_available():
        config.device = 'cuda:{}'.format(config.cuda_index)
    else:
        config.device = 'cpu'
    config.flush = not no_flush # if false, output scrolls (good for putting into .out file)
    config.visualise_overall_agent_results = False # for plotting
    config.load_model = load
    config.file_to_load_model = '/om/user/salford/models/' + load_model + '.pt'
    config.starting_episode_number = starting_ep # in case you want to resume training
    config.runs_per_agent = 1

    config.hyperparameters = {
        'DQN_Agents': {
            'learning_rate': 0.001,
            'batch_size': 128,
            'buffer_size': 100000,
            'ABCNN_hidden_units': 2048,
            'epsilon_decay_rate_denominator': 150,
            'discount_rate': 0.99,
            'incremental_td_error': 1e-8,
            'update_every_n_steps': 1,
            'gradient_clipping_norm': 5,
            'HER_sample_proportion': 0.8,
            'learning_iterations': 1,
            'clip_rewards': False,
            'net_type': net_type, # see create_NN method of Base_Agent.py
            'y_range': (-1, 10),
            'num_conv_layers': 3,
            # for FC
            'linear_hidden_units': [64]*2,
            'batch_norm': True,
            # for ResNet
            'num_blocks':num_blocks,
            'num_filters':num_filters,
            'device': config.device
        }
    }

    if env == 'reverse' or env == 'uncover':
        # since the important parameters aren't in the wrapper env
        config.info_string_to_log = info_maker.make_info_string(config, env.env)
    else:
        config.info_string_to_log = info_maker.make_info_string(config, env)

    AGENTS = [DQN_HER]

    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL parameters')

    parser.add_argument("--cuda_index", type=int, default='1',
            help="gpu device index")
    parser.add_argument("--id", type=int, default='1',
            help="id")
    parser.add_argument("--net_type", type=str, default='RNN',
            help="net type")
    args = parser.parse_args()

    cuda_index = args.cuda_index
    id = args.id

    run_random_search_forever(cuda_index, id, args.net_type)

