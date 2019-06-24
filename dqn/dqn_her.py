import sys
sys.path.append('../')
sys.path.append('../../Deep_RL_Implementations/')
import os
from gym.wrappers import FlattenDictWrapper
from agents.DQN_agents.DQN_HER import DQN_HER
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DQN import DQN
# to get the Deep RL modules accessible
from bitenvs.reverse_gym_env import ReverseGymEnv
import utilities.file_numberer as file_numberer

config = Config()
#config.seed = 1
str_len = 10
reverse_len = 3
reverse_offset = 1
num_obscured = 0

config.environment = ReverseGymEnv(str_len, reverse_len, reverse_offset,
        num_obscured)
filepath = f'/data/ReverseEnv_DQN_HER_{str_len}_{reverse_len}_{reverse_offset}_{num_obscured}'
filepath = file_numberer.get_unused_filepath(filepath, '.png', '.pkl',
        format_str='_({})')
config.file_to_save_data_results = filepath
config.file_to_save_results_graph = filepath

config.num_episodes_to_run = 15000
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {
    'DQN_Agents': {
        'learning_rate': 0.001,
        'batch_size': 128,
        'buffer_size': 100000,
        'epsilon_decay_rate_denominator': 150,
        'discount_rate': 0.999,
        'incremental_td_error': 1e-8,
        'update_every_n_steps': 1,
        'linear_hidden_units': [64, 64],
        'final_layer_activation': None,
        'y_range': (0, 1),
        'batch_norm': False,
        'gradient_clipping_norm': 5,
        'HER_sample_proportion': 0.8,
        'learning_iterations': 1,
        'clip_rewards': False
    }
}

if __name__== '__main__':
    AGENTS = [DQN_HER]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()


