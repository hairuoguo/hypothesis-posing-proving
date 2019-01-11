    

class BitStrEnv():
    '''
    should represent class of bit environments
    
    '''
    self.ep_type = None
    
    def __init__(self, str_len):
        self.str_len = str_len
        self.ep = None
        self.func_list = [
                
                #0 IDENTITY
            
                lambda x, y: x, 
                
                #1 AND

                lambda x, y: x & y, 

                #2 OR
                
                lambda x, y: x | y,

                #3 XOR

                lambda x, y: x ^ y,

                #4 ADD

                lambda x, y: x + y,

                #5 SUBTRACT_FROM

                lambda x, y: x - y,

                #6 FROM_SUBTRACT
            
                lambda x, y: y - x
            
            ]
   
     
    def start_ep(self):
        assert self.ep != None
        self.ep = self.ep_type(self.func_list, self.str_len)
        return self.ep 
         

class BitStrEpisode():
    '''
    should hold state of episode

    '''

    def __init__(self, actions_list, str_len):
        self.plain = None
        self.key = None
        self.encrypted = None
        self.state = None #current modified version of plain
        self.str_len = str_len
        self.actions_list = actions_list
        self.generate_strings()

    
    #return tuple (observation, reward) that is result of action
    def make_action(self, action_index):
        self.state = self.actions_list[action_index](self.state, self.key)
        return (self.get_obs(), self.get_reward())

    def get_reward(self):
        return int(self.state == self.encrypted)


    def get_obs(self):
        #TODO: determine observation

    def generate_strings(self):
        return
         
        
    
