from binary_gym_env import BinaryEnv

def test_gym_env(env, num_games=10):
    game_num = 0
    while game_num != num_games:
        print('New game')
        obs = env.reset()
        done = False
        print('action space: {}'.format(env.action_space))
        while not done:
            print(obs)
            a = input('choose an action: ')
            a = int(a)
            obs, reward, done, _ = env.step(a)
        game_num += 1
        print('Game over - I hope you won!')


if __name__ == '__main__':
    test_gym_env(BinaryEnv(5, 3), -1)

