import datetime

def make_info_string(*argv)
    '''
        Puts the values of the variables of each argument into a string form for
        easy recording of training run parameters.
    '''
    all_vars = []
    for arg in argv:
        all_vars += ['{0}: {1}'.format(k, v) for k, v in vars(arg).items()]

    s = '\n'.join(all_vars)
    return s
    
