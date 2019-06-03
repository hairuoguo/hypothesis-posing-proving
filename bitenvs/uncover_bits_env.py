import random
from types import MethodType
from bitenvs.reverse_env import *

class UncoverBitsEnv(ReverseEnv):

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured):
        super().__init__(str_len, reverse_len, reverse_offset, num_obscured)

    def start_ep(self):
        self.ep = UncoverBitsEpisode(self.actions_list, self.str_len, self.num_obscured, self.action_indices, self.reverse_len, self.reverse_offset)
        return self.ep


class UncoverBitsEpisode(ReverseEpisode):
    
    def __init__(self, actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset):
        super().__init__(actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset)
        def isEnd(self):
            return (self.entropy == 0.0)
        self.state.isEnd = MethodType(isEnd, self.state)


    def get_reward(self):
        return float(self.state.entropy == 0.0)



        

