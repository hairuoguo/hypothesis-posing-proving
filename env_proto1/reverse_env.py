import numpy as np
import random

class ReverseEnv:

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured):
        self.str_len = str_len #length of bitstring/bitarray
        self.reverse_len = reverse_len #length of each subsection that is reverse
        self.reverse_offset = reverse_offset #distance between start of each reversed section
        self.num_obscured = num_obscured #number of bits that are obscured
        self.actions_list = None
        self.indices = None
        self.__generate_func_list()
        self.__generate_indices()
        self.ep = None
    
    def __generate_func_list(self): #generates list of actions that reverse substrings
        self.actions_list = [(lambda x: (self.__reverse_substring(x, self.reverse_len, i))) for i in range(0, self.str_len-self.reverse_len, self.reverse_offset)]

    def __generate_indices(self):
        self.indices = [i for i in range(0, self.str_len-self.reverse_len, self.reverse_offset)]

    def __reverse_substring(self, bitarray, reverse_len, start_index): #function for reversing substrings
        bitarray[start_index:start_index+reverse_len] = bitarray[start_index:start_index+reverse_len][::-1]
        

    def start_ep(self):
        raise NotImplementedError


class ReverseEpisode:
    
    def __init__(self, actions_list, str_len, num_obscured):
        self.hidden_state = np.random.choice([1, 0], size=str_len) #actual state
        hidden = random.sample(range(str_len), num_obscured)
        self.hidden_mask = np.array([i for i in range(str_len) if i in hidden else 0])
        #self.hidden_mask = np.random.choice([1, 0], size=str_len, p=[num_obscured/str_len, 1-num_obscured/str_len]) #array indicating indices of obscured bits
        self.actions_list = actions_list 
        self.obs_state = None #state that is observable by agent (has obscured bits)
        self.target = np.copy(self.hidden_state) #target/goal array that we're tryin to arrive at
        self.num_obscured = num_obscured 
        self.str_len = str_len
        self.__update_obs_state() #generate initial observation
        self.generate_strings()
        self.possible_occluded_values = __self.__get_strings_d_away(np.zeros(str_len), np.sum(np.logical_and(self.hidden_state, self.hidden_mask)))
        self.possible_occluded_values = [self.__bitarray_to_int(bitarray) for bitarray in self.possible_occluded_values]
        self.__update_entropy()
    
    def __update_entropy(self):
        self.entropy = -1*math.log(len(self.possible_occluded_values), 2)
        
    def __bitarray_to_int(bitarray):
        return reduce(lambda x, y: x << 1 | y, bitarray, 0)
         
    def __update_obs_state(self):
        self.obs_state = np.copy(self.hidden_state)
        #print(self.hidden_mask)
        #print(np.argwhere(self.hidden_mask==1))
        self.obs_state[np.argwhere(self.hidden_mask!=0)] = -1 #obscured bits represented by -1

    def __concat_all_combinations(first_halves, second_halves):
        return [np.concatenate(a, b) for a in first_halves for b in second_halves]
        
    
    def __get_strings_d_away(bitarray, d):
        if distance == 0:
            return [bitstring]
        first_half = bitstring[:len(bitstring)//2]
        second_half = bitstring[len(btistring)//2:]
        strings = self.__concat_all_combinations(self.__get_strings_d_away(first_half, distance), second_half))
        for i in range(distance):
            if i >= len(first_half) and (distance - i) >= len(second_half):
                strings += self.__concat_all_combinations(self.__get_strings_d_away(first_half, i), self.__get_strings_d_away(second_half, distance-i)
        return strings

    def __update_poss_occluded_values(self):
        #get hamming distance from occluded bits only
        h_d = sum([self.target[i] ^ self.hidden_state[i] for i in range(str_len) if self.hidden_mask[i] != 0])
        target_bits = [(self.hidden_mask[i], self.target[i]) for i in range(str_len) if self.hidden_mask[i] != 0]
        target_bits = np.array([x[1] for x in target_bits.sort(key=lambda x:x[1]]))
        poss_strings = self.__get_strings_d_away(target_bits, h_d)
        poss_strings = [self.__bitarray_to_int(bitarray) for bitarray in poss_strings]
        self.possible_occluded_values = [x for x in self.possible_occluded_values if x in poss_strings]
    
    def __array_to_bitstring(bitarray):
        return "".join(bitarray.tolist())

    def make_action(self, action_index):
        self.actions_list[action_index](self.hidden_state) #perform reversal on hidden state
        self.actions_list[action_index](self.hidden_mask) #perform reversal on mask
        self.__update_poss_occluded_values()
        self.__update_entropy()
        self.__update_obs_state() #update observed state
        isEnd = self.hidden_state == self.target
        return (self.get_obs()[0], self.get_obs()[1], self.target_reached(), isEnd)

    def get_reward(self):
        return float(self.hidden_state == self.target)

    def target_reached(self):
        return self.hidden_state.tolist() == self.target.tolist()

    def get_obs(self):
        l1 = np.sum(np.abs(self.target - self.hidden_state)) 
        return self.obs_state, l1

    def generate_strings(self):
        #TODO generate problems that promote question-asking
        raise NotImplementedError 
    
        

    
