import sys
sys.path.append('../')
import os
from gym.wrappers import FlattenDictWrapper
from deep_rl.agents.DQN_agents.DQN_HER import DQN_HER
from deep_rl.agents.Trainer import Trainer
from deep_rl.utilities.data_structures.Config import Config
from deep_rl.agents.DQN_agents.DQN import DQN
# to get the Deep RL modules accessible
from bitenvs.reverse_gym_env import ReverseGymEnv
import deep_rl.utilities.file_numberer as file_numberer

config = Config()

str_len = 10
reverse_len = 3
reverse_offset = 1
num_obscured = 0

config.environment = ReverseGymEnv(str_len, reverse_len, reverse_offset,
        num_obscured)
data_dir = '/Users/alfordsimon/Python/hypothesis-posing-proving/dqn/data/reverse_env'
model_name = str.format('her_{0}_{1}_{2}_{3}', str_len,
        reverse_len, reverse_offset, num_obscured)

(config.file_to_save_data_results,
 config.file_to_save_model,
 config.file_to_save_results_graph,
 config.file_to_save_session_info) = file_numberer.get_unused_filepaths(data_dir,
         model_name)

config.seed = 1
config.num_episodes_to_run = 1000
config.starting_episode_number=1000
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True
config.seed = 1

config.load_model = True
config.file_to_load_model = data_dir + '/models/' + model_name + '_(1).pt'



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
#        'y_range': (0, 1),
        'y_range': (-1, str_len),
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


