import reverse_env as re


class SecondEpState(EpState):
    def __init__(self, *args):
        super().__init__(*args)
 

class SecondEnv(ReverseEnv):
    def __init__(self, *args):
        super().__init__(*args)

class SecondEnvEp(ReverseEpisode):

    #TODO: how to save actions (what should be format for representaitons?)
    
    def __init__(self, *args, **kwargs):
        self.action_limit = kwargs['action_limit']
        #agent should learn confidence for set of premises
        self.possible_premises = [] #initial state, target, actions should be able to be edited via composition
        self.active_hypothesis_eps = [] #currently active hypothesis eps and lists of action functions
        super().__init__(*args)

    def pose_subtask(self, initial_state, target, reverse_len, reverse_offset, hidden_indices):
        num_obscured = len(hidden_indices)
        hidden_state = initial_state
        env = ReverseEnv(len(target_string), reverse_len, reverse_offset, num_obscured)
        env.start_ep()
        state_args = {"hidden_state": initial_state,
            "hidden_indices": hidden_indices,
            "target": target,
            "num_obscured": num_obscured}
        env.ep.state = EpState(state_args)
        self.active_hypothesis_eps.append(env.ep)
        return env.ep

    def remove_subtask(self, ep):
        self.possible_premises.append(ep.path)
        self.active_hypothesis_eps.remove(ep)
         
        
    def make_action_subtask(self, action_index, ep):
            obs, reward, isEnd = ep.make_action(action_index)
            if isEnd:
                self.remove_subtask(ep)
        return obs, reward, isEnd 

    def make_action(self, action_index):
        self.state.make_action(self.actions_list[action_index])
        self.state.update_info()
        isEnd = self.state.isEnd()
        return (self.get_obs()[0], self.get_obs()[1], self.target_reached(), isEnd)

    def generate_strings(self, path_len_m, path_len_std):
        raise NotImplementedError 

    
