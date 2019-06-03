import argparse
import simple_reverse_env as se

parser = argparse.ArgumentParser(description='Bit reversal game')
parser.add_argument('-l', '--length', type=int, default=10,
                    help='Bitstring length (default: 10)')
parser.add_argument('-o', '--offset', type=int, default=1, 
                    help='Offset (default: 1)')
parser.add_argument('-of', '--obscured', type=int, default=1, 
                    help='Number of obscured bits (default: 1)')
parser.add_argument('-r', '--revlength', type=int, default=3, 
                    help='Length of reversals (default: 3)')

def print_bit_string(bits):
    bit_string = ""
    for i in bits:
        if i == -1:
            bit_string += "_"
        else:
            bit_string += str(i)
    print(bit_string)

if __name__ == '__main__':
    print("BIT REVERSAL GAME")
    print("\n")
    args = parser.parse_args()
    env = se.SimpleReverseEnv(args.length, args.revlength, args.offset, args.obscured)
    while True:
        again = input("Play an episode? (Y or N): ")
        print()
        if again == "N":
            break
        episode = se.SimpleReverseEpisode(env.actions_list, args.length, args.obscured)
        print("Bitsting length:", end = " ")
        print(args.length)
        print("Length of reversals:", end = " ")
        print(args.revlength)
        print("# of obscured bits:", end = " ")
        print(args.obscured)
        print("Offset:", end = " ")
        print(args.offset)
        while not(episode.target_reached()):
            print("Possible Indices to reverse:", end =" ")
            print(env.indices)
            print()
            print("Target:", end ="\t\t")
            print_bit_string(episode.target)
            print("Current State:", end ="\t")
            print_bit_string(episode.obs_state)
            print("Indicies:", end = "\t")
            for i in range(args.length):
                print(i%10, end = "")
            print()
            action = input("Select an index to reverse: ")
            episode.make_action(env.indices.index(int(action)))
        print("SUCCESS")
            
