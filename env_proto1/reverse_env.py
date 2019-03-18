import numpy as np
import random
import copy

class EpState:
    #should hold information about state of episode
    
    def __init__(self, *args, **kwargs):
        self.hidden_state = None
        self.hidden_indices = None
        self.target = None
        self.num_obscured = None
        self.__dict__.update(kwargs)
        self.hidden_mask = np.array([i for i in range(len(self.target)) if i in self.hidden_indices else 0])
        self.obs_state = None
        self.entropy = None
        self.occluded_bit_counts = None
        self.possible_occluded_values = None
        self.num_obscured = None
        
        self.__update_obs_state() #generate initial observation
        self.num_ones_in_occluded = np.sum(np.logical_and(self.hidden_state, self.hidden_mask))
        self.possible_occluded_values = __self.__get_strings_d_away(np.zeros(str_len), self.num_ones_in_occluded)
        self.possible_occluded_values = [self.__bitarray_to_int(bitarray) for bitarray in self.possible_occluded_values]
        self.__update_entropy()

    def update_entropy(self):
        self.entropy = math.log(len(self.possible_occluded_values), 2)
        self.occluded_bit_counts = np.zeros(self.num_obscured)
        for n in range(self.num_obscured):
            self.occluded_bit_counts = sum([m & 2**n for m in self.possible_occluded_values])
        
    def get_occluded_bit_ev(self, occluded_index):
        #occluded index is original index of occluded bit
        return self.occluded_bit_counts[self.hidden_indices.indexOf(occluded_index)]/len(self.possible_occluded_values)*self.num_ones_in_occluded


    def make_action(self, action_func):
        action_func(self.hidden_state)
        action_func(self.hidden_mask)
        self.__update_poss_occluded_values()
        self.update_entropy()
        self.__update_obs_state() #update observed state

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

    def isEnd(self):
        return self.hidden_state == self.target
    
    def __array_to_bitstring(bitarray):
        return "".join(bitarray.tolist())



class ReverseEnv:
    #should hold general information about environment

    def __init__(self, str_len, reverse_len, reverse_offset, num_obscured):
        self.str_len = str_len #length of bitstring/bitarray
        self.reverse_len = reverse_len #length of each subsection that is reverse
        self.reverse_offset = reverse_offset #distance between start of each reversed section
        self.num_obscured = num_obscured #number of bits that are obscured
        self.actions_list = None
        self.action_indices = None
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
    #should hold information about episode (str_len, reverse_len, etc)
    
    def __init__(self, actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset):
        
        self.reverse_len = None
        self.reverse_offset = None
        self.num_obscured = self.num_obscured
        self.action_indices = action_indices
        self.actions_list = actions_list 
        self.str_len = str_len
        self.state = None
        self.generate_strings()


    def make_action(self, action_index):
        self.state.make_action(self.actions_list[action_index])
        isEnd = self.state.isEnd()
        return (self.get_obs()[0], self.get_obs()[1], self.target_reached(), isEnd)

    def get_reward(self):
        return float(self.state.hidden_state == self.state.target)

    def target_reached(self):
        return self.state.hidden_state.tolist() == self.state.target.tolist()

    def get_obs(self):
        l1 = np.sum(np.abs(self.state.target - self.state.hidden_state)) 
        return self.obs_state, l1

    def __generate_hypothesis_ep(self, path_len, num_questions):
        question_indices = np.random.choice(range(path_len), num_questions)
        path = []
        actions = []
        path.append(self.state.hidden_state)
        for n in range(path_len):
            tried_actions = set()
            a_trial_state = copy.deepcopy(self.state)
            hidden_state = np.copy(self.hidden_state)
            constraint_satisfied = False
            while self.__bitarray_to_int(self.hidden_state) in states or not constraint_satisfied:
                if tried_actions == set(range(len(self.actions_list))):
                    return (path[0], path, question_indices)
                a = np.random.choice(range(len(self.actions_list)))
                tried_actions.add(a)
                a_trial_state.make_action(self.actions_list[a])
                if n in question_indices:
                    #get occlusions in action
                    action_index = self.action_indices[a]
                    action_occlusions = [i in self.hidden_mask[action_index:action_index + self.reverse_len] if i != 0]
                    a_ev_sum = sum([a_trial_state.get_occluded_bit_evs(bit_index) for bit_index in action_occlusions])
                    o_ev_sum = sum([self.state.get_occluded_bit_evs(bit_index) for bit_index in action_occlusions])
                    a_ev_change = a_ev_sum - o_ev_sum
                    for q in self.actions_list:
                        q_trial_state = copy.deepcopy(self.state)
                        if q == a: continue
                        q_trial_state.make_action(self.actions_list[q])
                        question_index = self.action_indices[q]
                        question_occlusions = [i in self.hidden_mask[question_index:question_index + self.reverse_len]]
                        entropy_indices = set(action_occlusions + question_occlusions)
                        q_ev_sum = sum([q_trial_state.get_occluded_bit_evs(bit_index) for bit_index in question_occlusions])
                        o_ev_sum = sum([self.state.get_occluded_bit_evs(bit_index) for bit_index in question_occlusions])
                        q_ev_change = q_ev_sum - o_ev_sum
                        if -1*q_ev_change + a_ev_sum > 1:
                            constraint_satisfied = True
            path.append(self.state.hidden_state)
            actions.append(a)
        return (target, path, question_indices)

    def generate_strings(self, path_len_m, path_len_std, num_qs_m, num_qs_std):
        path_len = math.ceil(np.random.normal(path_len_m, path_len_std)
        num_qs = math.floor(np.random.normal(num_qs_m, num_qs_std))
        hidden_state = np.random.choice([1, 0], size=self.str_len) #actual state
        hidden_indices = random.sample(range(self.str_len), self.num_obscured)
        hidden_mask = np.array([i for i in range(str_len) if i in self.hidden_indices else 0])
        self.state = EpState(hidden_state=hidden_state, hidden_indices=hidden_indices, hidden_mask=hidden_mask, num_obscured=self.num_obscured)
        orig_state = copy.deepcopy(self.state)
        target = self.__generate_hypothesis_ep(path_len, num_qs) #target/goal array that we're tryin to arrive at
        self.state = orig_state
        self.state.target = target
        self.__update_entropy()
        if np.array_equal(self.state.hidden_state, self.state.target): #if same, do over
            self.generate_strings()
    
        

    
