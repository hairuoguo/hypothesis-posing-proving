import sys
sys.path.append('../')
import os
#from gym.wrappers import FlattenDictWrapper
from deep_rl.agents.DQN_agents.DQN_HER import DQN_HER
from deep_rl.agents.DQN_agents.DQN_HER_alt import DQN_HER_alt
from deep_rl.agents.Trainer import Trainer
from deep_rl.utilities.data_structures.Config import Config
from deep_rl.agents.DQN_agents.DQN import DQN
from bitenvs.reverse_gym_env import ReverseGymEnv
import deep_rl.utilities.file_numberer as file_numberer
import argparse
import torch
import deep_rl.utilities.info_maker as info_maker
from pathlib import Path

parser = argparse.ArgumentParser(description='DQN-HER parameters')
parser.add_argument('-e', '--num_eps', default=10000, metavar='E', type=int, 
        help='number of episodes to run')
parser.add_argument('-n', '--str_len', default=10, metavar='N', type=int, 
        help='string length of Reverse Environment')
parser.add_argument('-r', '--reverse_len', default=3, metavar='R', type=int, 
        help='length of reversal operation')
parser.add_argument('--save_every', default=10000, metavar='N', type=int, 
        help='save data & model every _ episodes')
parser.add_argument('--path_len', default=5, metavar='L', type=int, 
        help='path length mean for Reverse Environment')
parser.add_argument('--file_name', default='', metavar='F', type=str, 
        help='file name to save data, model, info, plot with')
parser.add_argument('--info', default='', metavar='I', type=str, 
        help='info string for training run')
parser.add_argument("--cuda_index", default='1',metavar='I', type=int,
        help="gpu device index")
parser.add_argument("--num_blocks", default='1',metavar='I', type=int,
        help="num residual blocks for ResNet")
parser.add_argument("--num_filters", default='256',metavar='I', type=int,
        help="num_filters for ResNet")
parser.add_argument('--net_type', default='FC', metavar='N', type=str, 
        help='network type used by agent')
parser.add_argument("--no_save", help="don't save results", action="store_true")

args = parser.parse_args()
config = Config()

str_len = args.str_len
reverse_len = args.reverse_len
reverse_offset = 1
num_obscured = 0
path_len_mean = args.path_len
path_len_std = 0

env = ReverseGymEnv(str_len, reverse_len, reverse_offset, num_obscured,
        hypothesis_enabled=False, path_len_mean=path_len_mean,
        path_len_std=path_len_std, print_results=False)

config.save_every_n_episodes = args.save_every
config.save_results = not args.no_save
if not args.no_save:
    if len(args.file_name) > 0:
        model_name = args.file_name
    else:
        model_name = str.format('comp_{}_{}_{}_L{}', args.net_type, str_len, reverse_len,
                path_len_mean)

    data_dir = 'data/data'
    model_dir = '/om/user/salford/models' # because stored models take up lots of space
    plot_dir = 'data/plots'
    info_dir = 'data/info'

    (config.file_to_save_data_results,
     config.file_to_save_model,
     config.file_to_save_results_graph,
     config.file_to_save_session_info) = file_numberer.get_unused_filepaths(
             model_name, data_dir, model_dir, plot_dir, info_dir)

    # Immediately mark file as used so that other programs don't think it's untaken
    # in case we run lots of jobs at the same time
    Path(config.file_to_save_session_info).touch()

    if len(args.info) > 0:
        config.info = args.info
    else:
        config.info = 'comparing resnet and fc'

config.environment = env
config.no_random = False # if True, disables random actions but still trains
config.num_episodes_to_run = args.num_eps
config.cuda_index = args.cuda_index
if torch.cuda.is_available():
    config.device = 'cuda:{}'.format(config.cuda_index)
else:
    config.device = 'cpu'
config.flush = True # when logging performance each episode
config.visualise_overall_agent_results = False # for plotting
config.load_model = False
config.file_to_load_model = None
# config.starting_episode_number = 5 # in case you want to resume training

config.hyperparameters = {
    'DQN_Agents': {
        'learning_rate': 0.001,
        'batch_size': 128,
        'buffer_size': 100000,
        'epsilon_decay_rate_denominator': 150,
        'discount_rate': 0.999,
        'incremental_td_error': 1e-8,
        'update_every_n_steps': 1,
        'gradient_clipping_norm': 5,
        'HER_sample_proportion': 0.8,
        'learning_iterations': 1,
        'clip_rewards': False,
        'net_type': args.net_type, # see create_NN method of Base_Agent.py to see how used
        'y_range': (-1, str_len),
        # for CNN
        'num_conv_layers': 3,
        # for FC
        'linear_hidden_units': [1024]*3,
        'batch_norm': True,
        # for ResNet
        'num_blocks':args.num_blocks,
        'num_filters':args.num_filters
    }
}

config.info_string_to_log = info_maker.make_info_string(config, env.env)

if __name__== '__main__':
    AGENTS = [DQN_HER]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
