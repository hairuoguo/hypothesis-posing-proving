from bit_env import *
import random

         
        
class SimpleEnv(BitStrEnv):

    def __init__(self, str_len, rand_ratio=0.8):
        super().__init__(str_len)
        self.rand_ratio = rand_ratio
       
    
    def start_ep(self):
        self.ep = SimpleEp(self.func_list, self.str_len, self.rand_ratio)
        return self.ep
        
    
class SimpleEp(BitStrEpisode):

    def __init__(self, actions_list, str_len, rand_ratio):
        self.rand_ratio = rand_ratio
        super().__init__(actions_list, str_len)
        
    
    def generate_strings(self):
        rand = random.uniform(0, 1)
        if rand > self.rand_ratio:
            random.seed(42)
        else:
            random.seed()
        self.plain = random.getrandbits(self.str_len)
        self.key = random.getrandbits(self.str_len)
        self.encrypted, _ = self.actions_list[1](self.plain, self.key, [])
        self.encrypted, _ = self.actions_list[2](self.encrypted, self.key, [])
        self.state = self.plain

class AddEnv(SimpleEnv):

    def __init__(self, str_len):
        super().__init__(str_len)
       
    
    def start_ep(self):
        self.ep = AddEp(self.func_list, self.str_len)
        return self.ep

class OrEnv(SimpleEnv):

    def __init__(self, str_len):
        super().__init__(str_len)
       
    
    def start_ep(self):
        self.ep = OrEp(self.func_list, self.str_len)
        return self.ep

class AndOrEnv(SimpleEnv):

    def __init__(self, str_len, rand_ratio=0.5):
        super().__init__(str_len, rand_ratio)
       
    
    def start_ep(self):
        self.ep = AndOrEp(self.func_list, self.str_len, self.rand_ratio)
        return self.ep

class AddEp(SimpleEp):
    
    def __init__(self, actions_list, str_len):
        super().__init__(actions_list, str_len)

    
    def generate_strings(self):
        rand = random.uniform(0, 1)
        if rand < 0.5:
            random.seed(42)
        else:
            random.seed()
        self.plain = random.getrandbits(self.str_len)
        self.key = random.getrandbits(self.str_len)
        self.encrypted, _ = self.actions_list[1](self.plain, self.key, [])
        self.state = self.plain


class OrEp(SimpleEp): 
    
    def __init__(self, actions_list, str_len):
        super().__init__(actions_list, str_len)

    
    def generate_strings(self):
        rand = random.uniform(0, 1)
        if rand < 0.5:
            random.seed(42)
        else:
            random.seed()
        self.plain = random.getrandbits(self.str_len)
        self.key = random.getrandbits(self.str_len)
        self.encrypted, _ = self.actions_list[2](self.plain, self.key, [])
        self.state = self.plain

           
        
class AndOrEp(SimpleEp): 
    
    def __init__(self, actions_list, str_len, rand_ratio):
        super().__init__(actions_list, str_len, rand_ratio)

    
    def generate_strings(self):
        rand = random.uniform(0, 1)
        if rand > self.rand_ratio:
            random.seed(42)
        else:
            random.seed()
        self.plain = random.getrandbits(self.str_len)
        self.key = random.getrandbits(self.str_len)
        random.seed()
        rand = random.uniform(0, 1)
        if rand < 0.5:
            self.encrypted, _ = self.actions_list[1](self.plain, self.key, [])
        else:
            self.encrypted, _ = self.actions_list[2](self.plain, self.key, [])
        self.state = self.plain
