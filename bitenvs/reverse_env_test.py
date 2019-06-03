from reverse_env import *



if __name__ == "__main__":
    env = ReverseEnv(10, 3, 1, 5)
    env.start_ep()
    print(env.ep.state.target)
    print(env.ep.state.hidden_state)
    print(env.ep.state.hidden_mask)
    print(env.ep.make_action(len(env.ep.actions_list)-1))
    print(env.ep.hypothesis_on)
