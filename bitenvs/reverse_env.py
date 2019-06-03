import numpy as np
import types
from functools import reduce
import functools
import math
import random
import copy

class EpStats:
    #holds information about episode, to be returned after episode terminated
    
    def __init__(self, *args, **kwargs):
        self.path = []
        self.entropy_decrease = []
        self.max_poss_entropy_decrease = []
        self.__dict__.update(kwargs)

class EpState:
    #should hold information about state of episode
    
    def __init__(self, *args, **kwargs):
        self.hidden_state = None
        self.hidden_indices = None
        self.target = None
        self.num_obscured = None
        self.__dict__.update(kwargs)
        self.str_len = len(self.hidden_state)
        self.hidden_mask = None
        self.make_hidden_mask()
        self.obs_state = None
        self.entropy = None
        self.occluded_bit_counts = None
        self.possible_occluded_values = None
        
        self.__update_obs_state() #generate initial observation
        self.num_ones_in_occluded = np.sum(np.logical_and(self.hidden_state, self.hidden_mask))
        self.possible_occluded_values = self.__get_strings_d_away(np.zeros(self.num_obscured), self.num_ones_in_occluded)
        self.possible_occluded_values = [self.bitarray_to_int(bitarray) for bitarray in self.possible_occluded_values]
        self.update_entropy()

    def make_hidden_mask(self):
        self.hidden_mask = np.array([i if i in self.hidden_indices else 0 for i in range(self.str_len)])


    def update_entropy(self):
        self.entropy = math.log(len(self.possible_occluded_values), 2)
        self.occluded_bit_counts = np.zeros(self.num_obscured)
        for n in range(self.num_obscured):
            self.occluded_bit_counts[n] = sum([bool(m & (1 << n)) for m in self.possible_occluded_values])
        

    def get_occluded_bit_ev(self, occluded_index):
        #occluded index is original index of occluded bit
        return self.occluded_bit_counts[self.hidden_indices.indexOf(occluded_index)]/len(self.possible_occluded_values)*self.num_ones_in_occluded


    def make_action(self, action_func):
        action_func(self.hidden_state)
        action_func(self.hidden_mask)
        self.__update_obs_state() #update observed state

    def update_info(self):
        self.__update_poss_occluded_values()
        self.update_entropy()

    def bitarray_to_int(self, bitarray):
        return functools.reduce(lambda x, y: x << 1 | int(y), bitarray, 0)
         
    def __update_obs_state(self):
        self.obs_state = np.copy(self.hidden_state)
        self.obs_state[np.argwhere(self.hidden_mask!=0)] = -1 #obscured bits represented by -1

    def __concat_all_combinations(self, first_halves, second_halves):
        return [np.concatenate((a, b)) for a in first_halves for b in second_halves]

    def __get_strings_d_away(self, bitarray, d):
        if d == 0:
            return [bitarray]
        if len(bitarray) == 1 and d == 1:
            return [np.abs(bitarray - np.array([1.]))]
        first_half = bitarray[:len(bitarray)//2]
        second_half = bitarray[len(bitarray)//2:]
        #strings = self.__concat_all_combinations(self.__get_strings_d_away(first_half, d), second_half)
        strings = []
        for i in range(d+1):
            if i <= len(first_half) and (d - i) <= len(second_half):
                first_halves = self.__get_strings_d_away(first_half, i)
                second_halves = self.__get_strings_d_away(second_half, d-i)
                #strings += self.__concat_all_combinations(self.__get_strings_d_away(first_half, i), self.__get_strings_d_away(second_half, d-i))
                strings += self.__concat_all_combinations(first_halves, second_halves)
        return strings
        
    def __update_poss_occluded_values(self):
        #get hamming distance from occluded bits only
        h_d = sum([self.target[i] ^ self.hidden_state[i] for i in range(self.str_len) if self.hidden_mask[i] != 0])
        target_bits = [(self.hidden_mask[i], self.target[i]) for i in range(self.str_len) if self.hidden_mask[i] != 0]
        target_bits = np.array([x[1] for x in sorted(target_bits, key=lambda x: x[0])])
        poss_strings = self.__get_strings_d_away(target_bits, h_d)
        poss_strings = [self.bitarray_to_int(bitarray) for bitarray in poss_strings]
        self.possible_occluded_values = [x for x in self.possible_occluded_values if x in poss_strings]

    def isEnd(self):
        return np.array_equal(self.hidden_state, self.target)
    
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
        self.actions_list = [functools.partial(self.__reverse_substring, reverse_len=self.reverse_len, start_index=i) for i in range(0, self.str_len-self.reverse_len + 1, self.reverse_offset)]

    def __generate_indices(self):
        self.action_indices = [i for i in range(0, self.str_len-self.reverse_len, self.reverse_offset)]

    def __reverse_substring(self, bitarray, reverse_len, start_index): #function for reversing substrings
        bitarray[start_index:start_index+reverse_len] = bitarray[start_index:start_index+reverse_len][::-1]

    def hypothesis_enable_ep(self, ep):
        #takes episode and modifies it to allow for hypothesis-posing
        ep.hypothesis_on = False
        ep.actions_list.append((lambda: None)) #empty function to represent hypothesis-posing
        ep.branch_state = copy.deepcopy(ep.state)
        def make_action_wrapper(make_action_function):
            def new_make_action(self, action_index):
                if action_index == len(self.actions_list) - 1: 
                    if self.hypothesis_on:
                        self.hypothesis_on = False
                        self.state = copy.deepcopy(self.branch_state)
                    else:
                        self.hypothesis_on = True
                        self.branch_state = copy.deepcopy(self.state)
                else: 
                    make_action_function(action_index)
            return new_make_action
        ep.make_action = types.MethodType(make_action_wrapper(ep.make_action), ep)
        

    def start_ep(self):
        self.ep = ReverseEpisode(self.actions_list, self.str_len, self.num_obscured, self.action_indices, self.reverse_len, self.reverse_offset)
        if self.num_obscured > 0:
            self.hypothesis_enable_ep(self.ep)
        return self.ep

    def seed(self, seed):
        np.random.seed(seed)
         


class ReverseEpisode:
    #should hold information about episode (str_len, reverse_len, etc)

    
    def __init__(self, actions_list, str_len, num_obscured, action_indices, reverse_len, reverse_offset):
        
        self.reverse_len = None
        self.reverse_offset = None
        self.num_obscured = num_obscured
        self.action_indices = action_indices
        self.actions_list = actions_list 
        self.str_len = str_len
        self.state = None
        self.generate_strings(5, 0.5, 2, 0)
        self.stats = EpStats() 


    def make_action(self, action_index):
        curr_entropy = self.state.entropy
        self.stats.max_poss_entropy_decrease.append(self.get_max_poss_entropy_decrease)
        self.state.make_action(self.actions_list[action_index])
        self.state.update_info()
        isEnd = self.state.isEnd()
        if self.stats.path:
            self.stats.path.append(copy.deepcopy(self.state))
        self.stats.entropy_decrease.append(curr_entropy - self.state.entropy)
        return (self.get_obs()[0], self.get_obs()[1], self.get_reward(), isEnd)

    def get_reward(self):
        if np.array_equal(self.state.hidden_state, self.state.target):
            return 1.
        else:
            return 0.

    '''
    def target_reached(self):
        return self.state.hidden_state.tolist() == self.state.target.tolist()
    '''

    def get_obs(self):
        l1 = np.sum(np.abs(self.state.target - self.state.hidden_state)) 
        return (np.concatenate((self.state.obs_state, self.state.target)), l1)

    def get_max_poss_entropy_decrease(self):
        curr_entropy = self.state.entropy
        max_decrease = 0
        for action_func in self.actions_list:
            copy_state = copy.deepcopy(self.state)
            copy_state.make_action(action_func)
            copy_state.update_info()
            max_decrease = max(max_decrease, curr_entropy - copy_state.entropy)
            del copy_state
        return max_decrease
            
   

    def __generate_hypothesis_ep(self, path_len, num_questions):
        #method generates target from self.state with initial hidden string
        question_indices = np.random.choice(range(path_len), num_questions, replace=False)#indices in path where we want to promote question-asking
        path = [] #list of states representing path that will be answer
        actions = [] #list of actions associated with path
        hidden_states = set() #states that algorithm has reached before
        hidden_states.add(self.state.bitarray_to_int(self.state.hidden_state))
        path.append(copy.deepcopy(self.state))
        
        for n in range(path_len):
            #try random action, until we get an action that doesn't lead to path seen before
            tried_actions = set() #actions we've already tried for this index
            a_trial_state = copy.deepcopy(self.state) #copy that we will test actions on
            while (a_trial_state.bitarray_to_int(a_trial_state.hidden_state)) in hidden_states:
                a_trial_state = copy.deepcopy(self.state)
                if tried_actions == set(range(len(self.actions_list))):
                    #if we've already tried all actions, return values that will make generate_strings call itself again.
                    return (path[0].hidden_state, path, question_indices)
                a = np.random.choice(range(len(self.actions_list)))
                tried_actions.add(a)
                a_trial_state.make_action(self.actions_list[a])
            actions.append(a)
            self.state = a_trial_state
            hidden_states.add(a_trial_state.bitarray_to_int(a_trial_state.hidden_state))
        self.state = copy.deepcopy(path[0])
        self.state.target = np.copy(a_trial_state.hidden_state) #provisional target 
        self.state.hidden_indices = [i for i in self.state.hidden_mask if i != 0] #reset hidden inidices to be relative to target (were originally relative to initial hidden state)
        self.state.make_hidden_mask()
        path[0].target = np.copy(self.state.target) 
        for action in actions:
            #generate path since we now have target and can properly generate entropy info for each step in path (since entropy calculation depends on target)
            self.state.make_action(self.actions_list[action])
            self.state.update_info()
            path.append(copy.deepcopy(self.state))
        for question_index in question_indices:
            #if action is where we want to incentivize question-asking
            action_taken = actions[question_index]
            action_entropy_decrease = path[question_index].entropy - path[question_index+1].entropy
            #entropy decrease from action taken
            if self.num_obscured > 0:
                constraint_satisfied = False
            else:
                constraint_satisfied = True
            for q in range(len(self.actions_list)):
                trial_q_state = copy.deepcopy(path[question_index])
                if q == action_taken:
                    continue 
                trial_q_state.make_action(self.actions_list[q])
                trial_q_state.update_info()
                question_entropy_decrease = path[question_index].entropy - trial_q_state.entropy
                #entropy decrease from each question
                if question_entropy_decrease - action_entropy_decrease > 1:
                    #if there exists a question that satisfies this constraint 
                    constraint_satisfied = True
                    break
            if not constraint_satisfied:
                #if not satisfied, reset initial state of episode try to generate different path
                self.state = copy.deepcopy(path[0])
                return (path[0].hidden_state, path, question_indices)
                #return self.__generate_hypothesis_ep(path_len, num_questions)
        return (self.state.target, path, question_indices)

    def generate_strings(self, path_len_m, path_len_std, num_qs_m, num_qs_std, hidden_state=None):
        path_len = math.ceil(np.random.normal(path_len_m, path_len_std))
        num_qs = math.floor(np.random.normal(num_qs_m, num_qs_std))
        if hidden_state == None:
            hidden_state = np.random.choice([1, 0], size=self.str_len) #actual state
        hidden_indices = random.sample(range(self.str_len), self.num_obscured) #indicies of hidden states
        hidden_mask = np.array([i if i in hidden_indices else 0 for i in range(self.str_len)])
        self.state = EpState(hidden_state=hidden_state, hidden_indices=hidden_indices, hidden_mask=hidden_mask, num_obscured=self.num_obscured)
        orig_state = copy.deepcopy(self.state)
        target, path, question_indices = self.__generate_hypothesis_ep(path_len, num_qs) #target/goal array that we're tryin to arrive at
        self.state = orig_state
        if np.array_equal(self.state.hidden_state, target): #if same, do over
            self.generate_strings(path_len_m, path_len_std, num_qs_m, num_qs_std)
        else:
            self.state.target = target
            self.state.update_info()
    
    ''' 
    def __generate_hypothesis_ep(self, path_len, num_questions):
        for n in range(len(self.actions_list)):
            trial_state = copy.deepcopy(self.state)
            trial_state.make_action(self.actions_list[n])
            print(n, self.state.hidden_state, trial_state.hidden_state)
    '''
        

    
