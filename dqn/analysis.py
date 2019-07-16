import pickle
import numpy as np
import matplotlib.pyplot as plt
env_name = 'cnn_10_3'

with open('data/' + env_name + '.pkl','rb') as f:
    data = pickle.load(f)
    print(data.keys)
    data = data['DQN_HER'] # only one agent 
    print(len(data))
    data = data[0] # only one run for this agent
    print(len(data))
    ep_lens, avg_ep_lens, _, _, secs_taken = data
    last_eps = ep_lens[-1000:]
    print(ep_lens[100:])
    print(last_eps[-10:])

