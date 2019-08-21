# make plots for any training runs which have a /data/model_name.pkl but not a
# /plots/model_name.png
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import pickle
import numpy as np

def save_plot(env_name, path):
    data_path = path + 'data/' + env_name + '.pkl'
    plot_path = path + 'plots/' + env_name + '.png'
    print('Plotting {}'.format(env_name))
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        s = 'DQN_HER'
        if s not in data.keys():
            s = 'DQN-HER'
        if s not in data.keys():
            s = 'DQN'

        data = data[s] # only one agent 
        data = data[0] # only one run for this agent
        if len(data) == 5:
            ep_lens, avg_ep_lens, _, _, secs_taken = data
            max_steps = max(ep_lens)
            solved = np.array(ep_lens) < max_steps
            percent_solved = [np.sum(solved[i-50:i]) / 50 for i in
                    range(len(solved))]
        else: # have rolling percent solved!
            ep_lens, avg_ep_lens, percent_solved, _, _, secs_taken = data

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(avg_ep_lens, color='blue', linewidth=.3)
        plt.title('rolling 50-avg episode lengths ' + env_name)
        plt.xlabel('episode number')
        plt.ylabel('rolling 50-avg ep. length')
        plt.grid(axis='y')
        plt.subplot(2, 1, 2)
        plt.plot(percent_solved, color='orange', linewidth=.3)
        plt.title('Percent solved ' + env_name)
        plt.xlabel('episode number')
        plt.ylabel('percent solved before ending')
        plt.grid(axis='y')
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)

path = '/Users/alfordsimon/Python/hypothesis-posing-proving-data/'
data_path = path + 'data/'
plot_path = path + 'plots/'
files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

plots = []
for file in files:
    env_name = file[:-4] # remove extension
    if not isfile(plot_path + env_name + '.png'):
        save_plot(env_name, path)
        plots.append(env_name)

print('Made {} plots.'.format(len(plots)))
