import pickle
import numpy as np
import matplotlib.pyplot as plt

env_name = 'cnn_5_2_(9)'

with open('data/' + env_name + '.pkl','rb') as f:
    data = pickle.load(f)
    print(data.keys)
    data = data['DQN_HER'] # only one agent 
    print(len(data))
    data = data[0] # only one run for this agent
    print(len(data))
    ep_lens, avg_ep_lens, _, _, secs_taken = data
    print('steps: {}'.format(len(ep_lens)))
    print(secs_taken)
    last_eps = ep_lens[-1000:]
    print(avg_ep_lens[-10:])

