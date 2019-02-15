import numpy as np

class ReverseEnv:

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured):
        self.str_len = str_len
        self.reverse_len = reverse_len
        self.reverse_offset = reverse_offset
        self.num_obscured = num_obscured
        self.actions_list = None
        self.__generate_func_list()
        self.ep = None
    
    def __generate_func_list(self):
        self.actions_list = [(lambda x: (self.__reverse_substring(x, self.reverse_len, i))) for i in range(0, self.str_len-self.reverse_len, self.reverse_offset)] 

    def __reverse_substring(self, bitarray, reverse_len, start_index):
        bitarray[start_index:start_index+reverse_len] = bitarray[start_index:start_index+reverse_len][::-1]
        

    def start_ep(self):
        raise NotImplementedError


class ReverseEpisode:
    
    def __init__(self, actions_list, str_len, num_obscured):
        self.hiddenstate = np.random.choice([1, 0], size=str_len)
        self.hidden_mask = np.random.choice([1, 0], size=str_len, p=[num_obscured/str_len, 1-num_obscured/str_len])
        self.actions_list = actions_list 
        self.obs_state = None
        self.target = np.copy(self.hidden_state)
        self.num_obscured = num_obscured
        self.str_len = str_len
        self.__update_obs_state()
        self.generate_strings()
         
    def __update_obs_state(self):
        self.obs_state = np.copy(self.hidden_state)
        self.obs_state[np.argwhere(self.hidden_mask==1)] = -1
    
    def __array_to_bitstring(bitarray):
        return "".join(bitarray.tolist()) 

    def make_action(self, action_index):
        self.hidden_state = self.actions_list[action_index](self.hidden_state)
        self.hidden_mask = self.actions_list[action_index](self.hidden_mask)
        isEnd = self.hidden_state == self.target
        return (self.get_obs()[0], self.get_obs()[1], self.get_reward(), isEnd)

    def get_reward(self):
        return float(self.hidden_state == self.target)

    def get_obs(self): 
        l1 = np.sum(np.abs(self.target - self.hidden_state)) 
        return self.obs_state, l1

    def generate_strings(self):
        raise NotImplementedError 
    
        

    
