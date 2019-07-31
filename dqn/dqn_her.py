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

parser = argparse.ArgumentParser(description='DQN-HER parameters')
parser.add_argument('-e', '--num_eps',type=int, default=10000,  
        help='number of episodes to run')
parser.add_argument('-n', '--str_len',type=int, default=10,  
        help='string length of Reverse Environment')
parser.add_argument('-r', '--reverse_len',type=int, default=3,  
        help='length of reversal operation')
parser.add_argument('-se', '--save_every',type=int, default=25000,  
        help='save data & model every _ episodes')
parser.add_argument('-l', '--path_len',type=int, default=3,  
        help='path length mean for Reverse Environment')
parser.add_argument('-f', '--file_name',type=str, default='',  
        help='file name to save data, model, info, plot with')
parser.add_argument('-i', '--info',type=str,
        default='',  
        help='info string for training run')
parser.add_argument("--cuda_index", type=int, default='1',
        help="gpu device index")
parser.add_argument("--num_blocks", type=int, default='1',
        help="num residual blocks for ResNet")
parser.add_argument("--num_filters", type=int, default='10',
        help="num_filters for ResNet")
parser.add_argument('-t', '--net_type', type=str, default='FC',
        help='network type used by agent')
parser.add_argument('-lm', "--load_model", type=str, 
        default='AC_7_3',
        help="model file to load")
parser.add_argument("--starting_ep", type=int, default='0',
        help="starting episode for resuming training")
parser.add_argument('-ns', "--no_save", action='store_true', 
        help="don't save results")
parser.add_argument('-nf', "--no_flush", action='store_true', 
        help="don't flush output")
parser.add_argument("--load", action="store_true",
        help="load model")


args = parser.parse_args()
config = Config()

str_len = args.str_len
reverse_len = args.reverse_len
# config.reverse_len = args.reverse_len # needed for all_conv.
config.reverse_len = 1 # needed for all_conv.
reverse_offset = 1
num_obscured = 0
path_len_mean = args.path_len
path_len_std = 0

# env = ReverseGymEnv(str_len, reverse_len, reverse_offset, num_obscured,
#         hypothesis_enabled=False, path_len_mean=path_len_mean,
#         path_len_std=path_len_std, print_results=False)
env = BinaryEnv(str_len, path_len_mean) # path_len is num. bits flipped to 1
#env = Bit_Flipping_Environment(environment_dimension=str_len)

config.save_every_n_episodes = args.save_every
config.save_results = not args.no_save

data_dir = '/om/user/salford/data'
model_dir = '/om/user/salford/models'
plot_dir = '/om/user/salford/plots'
info_dir = '/om/user/salford/info'

if not args.no_save:
    if len(args.file_name) > 0:
        model_name = args.file_name
    else:
        model_name = str.format('{}_{}_{}_L{}', args.net_type,
                str_len, reverse_len, path_len_mean)

    (config.file_to_save_data_results,
     config.file_to_save_model,
     config.file_to_save_results_graph,
     config.file_to_save_session_info) = file_numberer.get_unused_filepaths(
             model_name, data_dir, model_dir, plot_dir, info_dir)

    # Immediately mark file as used so that other programs don't think it's untaken
    # in case we run lots of jobs at the same time
    Path(config.file_to_save_session_info).touch()

    config.info = args.info

config.environment = env
config.no_random = False # if True, disables random actions but still trains
config.num_episodes_to_run = args.num_eps
config.cuda_index = args.cuda_index
if torch.cuda.is_available():
    config.device = 'cuda:{}'.format(config.cuda_index)
else:
    config.device = 'cpu'
config.flush = not args.no_flush # if false, output scrolls (good for putting into .out file)
config.visualise_overall_agent_results = False # for plotting
config.load_model = args.load
config.file_to_load_model = '/om/user/salford/models/' + args.load_model + '.pt'
config.starting_episode_number = args.starting_ep # in case you want to resume training

config.hyperparameters = {
    'DQN_Agents': {
        'learning_rate': 0.001,
        'batch_size': 5,
        'buffer_size': 100000,
        'epsilon_decay_rate_denominator': 150,
        'discount_rate': 0,
        'incremental_td_error': 1e-8,
        'update_every_n_steps': 1,
        'gradient_clipping_norm': 5,
        'HER_sample_proportion': 0.8,
        'learning_iterations': 1,
        'clip_rewards': False,
        'net_type': args.net_type, # see create_NN method of Base_Agent.py to see how used
        # assuming std is zero, this is good. if not may need three std higher
        # to cover almost all possible values.
        'y_range': (-1, 2*args.str_len + 1), 
        'num_conv_layers': 3,
        # for FC
        'linear_hidden_units': [64]*2,
        'batch_norm': True,
        # for ResNet
        'num_blocks':args.num_blocks,
        'num_filters':args.num_filters
    }
}

# config.info_string_to_log = info_maker.make_info_string(config, env.env)

if __name__== '__main__':
    AGENTS = [DQN, DQN_HER]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
