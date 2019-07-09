import datetime

def make_info_string(config, reverse_env):
    s = '\n'.join((
          [str.format('{0}: {1}',k,v) for k, v in vars(config).items()]
        + [str.format('{0}: {1}',k,v) for k, v in vars(reverse_env).items()]
        + [str(datetime.datetime.now())]
        ))
    return s
    
