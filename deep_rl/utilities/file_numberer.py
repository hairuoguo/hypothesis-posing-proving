import os
import numpy as np

def get_unused_filepaths(data_dir, model_name, format_str='_({})'):
    folders = ['data','models','plots','info']
    exts = ['.pkl', '.pt', '.png', '.txt']
    i = 0
    i_str = ''
    while np.any([os.path.isfile( data_dir + '/' + folder + '/' + model_name
        + i_str + ext) for folder, ext in zip(folders, exts)]):
        i += 1
        i_str = str.format(format_str, i)

    return tuple(data_dir + '/' + folder + '/' + model_name + i_str + ext for folder, ext in zip(folders, exts))



