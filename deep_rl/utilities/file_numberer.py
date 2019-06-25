import os
import numpy as np

def get_unused_filepath(filepath, *argv, format_str='_({})'):
    i = 0
    i_str = ''
    while np.any([os.path.isfile(filepath + i_str + ext) for ext in argv]):
        i += 1
        i_str = str.format(format_str, i)

#    return tuple(filepath + i_str + ext for ext in argv)
    return filepath + i_str

