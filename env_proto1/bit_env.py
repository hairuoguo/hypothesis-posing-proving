class BitStrEnv:
    '''
    should represent class of bit environments
    
    '''
    
    def __init__(self, str_len):
        self.str_len = str_len
        self.ep = None
        self.func_list = [
                
                #0 IDENTITY
            
                lambda x, y, z: (x, z.append("Id")),
                
                #1 AND

                lambda x, y, z: (x & y, z.append("And")),

                #2 OR
                
                lambda x, y, z: (x | y, z.append("Or")),

                #3 XOR

                lambda x, y, z: (x ^ y, z.append("XOr")),

                #4

                lambda x, y, z: (x, z.pop())
                
            ]
   
     
    def start_ep(self):
        return
         

class BitStrEpisode:
    '''
    should hold state of episode

    '''

    def __init__(self, actions_list, str_len):
        self.plain = None
        self.key = None
        self.encrypted = None
        self.state = None #current modified version of plain
        self.previous_states = [] #Store previous states for backtracking
        self.str_len = str_len
        self.actions_list = actions_list
        self.actions_taken = []
        self.generate_strings()

    
    #return tuple (observation, reward) that is result of action
    def make_action(self, action_index):
        last_action = None
        if not (action_index == 4 and not self.actions_taken):
            #Update state and get last action if backtracking
            self.state, last_action = self.actions_list[action_index](self.state, self.key, self.actions_taken)
        if last_action is not None:
            self.backtrack(last_action)
        else:
            self.backtrack_flag = 0
        self.previous_states.append(self.state)
        isEnd = self.state == self.encrypted
        return (self.get_obs(), self.get_reward(), isEnd)

    def backtrack(self, action):
        self.state = self.previous_states.pop()

    def get_reward(self):
        return float(self.state == self.encrypted)


    def get_plain(self):
        return ('{0:0%sb}' % (self.str_len, )).format(self.plain)

    def get_key(self):
        return ('{0:0%sb}' % (self.str_len, )).format(self.key)

    def get_encrypted(self):
        return ('{0:0%sb}' % (self.str_len, )).format(self.encrypted)

    def get_state(self):
        return ('{0:0%sb}' % (self.str_len, )).format(self.state)

    def get_obs(self):
        return self.get_state()

    def generate_strings(self):
        return
