# make plots for any training runs which have a /data/model_name.pkl but not a
# /plots/model_name.png
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import pickle

def save_plot(env_name, save_path):
    print('Plotting {}'.format(env_name))
    with open('data/' + env_name + '.pkl','rb') as f:
        data = pickle.load(f)
        data = data['DQN_HER'] # only one agent 
        data = data[0] # only one run for this agent
        ep_lens, avg_ep_lens, _, _, secs_taken = data
        plt.clf()
        plt.plot(avg_ep_lens)
        plt.title('rolling 50-avg episode lengths ' + env_name)
        plt.xlabel('episode no.')
        plt.ylabel('rolling 50-avg ep. length')
        plt.savefig(save_path)

path = 'data/'
files = [f for f in listdir(path) if isfile(join(path, f))]

plots = []
for file in files:
    env_name = file[:-4]
    plot_file = 'plots/' + env_name + '.png'
    if not isfile(plot_file):
        save_plot(env_name, plot_file)
        plots.append(env_name)

print('Made {} plots.'.format(len(plots)))
