class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = 1
        self.randomise_random_seed = True

        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.starting_episode_number = None

        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.file_to_save_model = None
        self.file_to_save_session_info = None
        self.file_to_load_model = None
        self.overwrite_existing_results_file = None

        self.runs_per_agent = 1
        self.visualise_overall_agent_results = True
        self.visualise_individual_results = False
        self.hyperparameters = None
        self.standard_deviation_results = 1.0
        self.show_solution_score = False
        self.debug_mode = False
        self.log_training = False
        self.no_random = False

        self.load_model = False
        self.save_results = True
        self.save_every_n_episodes = None

        self.run_info = None
        self.info_string_to_log = ""
        self.flush = False

        self.cuda_index = 0
        self.device = 'cuda:0'



