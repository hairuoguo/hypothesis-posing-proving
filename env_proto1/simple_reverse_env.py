import random
from reverse_env import *

class SimpleReverseEnv(ReverseEnv):

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured):
        super().__init__(str_len, reverse_len, reverse_offset, num_obscured)

    def start_ep(self):
        self.ep = SimpleReverseEpisode(self.actions_list, self.str_len, self.num_obscured)
        return self.ep



class SimpleReverseEpisode(ReverseEpisode):
    
    def __init__(self, actions_list, str_len, num_obscured):
        super().__init__(actions_list, str_len, num_obscured)

    def generate_strings(self):
        rand_act = random.choice(self.actions_list) #choose random action to perform
        rand_act(self.target)
        

