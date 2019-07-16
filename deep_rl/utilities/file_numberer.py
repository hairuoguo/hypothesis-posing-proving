import os
import numpy as np

def get_unused_filepaths(model_name, format_str='_({})'):
#    folders = ['data/data', '/om/user/salford/models'
    folders = ['data/data', '/Users/alfordsimon/data'
            ,'data/plots', 'data/info']
    exts = ['.pkl', '.pt', '.png', '.txt']
    i = 0
    i_str = ''
    while np.any([os.path.isfile(folder + '/' + model_name
        + i_str + ext) for folder, ext in zip(folders, exts)]):
        i += 1
        i_str = str.format(format_str, i)

    return tuple(folder + '/' + model_name + i_str + ext for folder, ext in zip(folders, exts))



