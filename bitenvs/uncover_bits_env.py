import random
from types import MethodType
from bitenvs.reverse_env import *

class UncoverBitsEnv(ReverseEnv):

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured, hypothesis_enabled, path_len_mean, path_len_std):
        super().__init__(str_len, reverse_len, reverse_offset, num_obscured, hypothesis_enabled, path_len_mean, path_len_std)

    def start_ep(self):
        self.ep = UncoverBitsEpisode(self.actions_list, self.str_len, self.num_obscured, self.action_indices, self.reverse_len, self.reverse_offset, self.path_len_mean, self.path_len_std)
        if self.ep.state.entropy == 0.0:
            self.ep = self.start_ep()
        return self.ep


class UncoverBitsEpisode(ReverseEpisode):
    
    def __init__(self, actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset, path_len_mean, path_len_std):
        super().__init__(actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset, path_len_mean, path_len_std)
        def isEnd(self):
            return (self.entropy == 0.0)
        self.state.isEnd = MethodType(isEnd, self.state)


    def get_reward(self):
        return float(self.state.entropy == 0.0)



        

