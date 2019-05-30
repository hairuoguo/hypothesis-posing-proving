from bit_env import *
import random

         
        
class IdentityEnv(BitStrEnv):

    def __init__(self, str_len):
        super().__init__(str_len)
        
    
    def start_ep(self):
        self.ep = IdentityEp(self.func_list, self.str_len)
        return self.ep
        
    
class IdentityEp(BitStrEpisode):

    def __init__(self, actions_list, str_len):
        super().__init__(actions_list, str_len)
        
    
    def generate_strings(self):
        self.plain = random.getrandbits(self.str_len)
        self.key = random.getrandbits(self.str_len)
        self.encrypted = self.plain
        self.state = self.plain

           
        
