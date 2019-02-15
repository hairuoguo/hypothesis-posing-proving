

* identity done
* used Reinforce

* problem in second most simple task
    * ciphertext = Op_2( Op_1( plainttext, key ), key )
        * Op1 = And
        * Op2 = Or
        * its not invariant to order
    * actions = {backtrack, identity, OR, AND, XOR}
    * perform the series of operations to arrive at the ciphertext
* Q: number of strings 
    * different for each episode
        * episode, actions until it gets a reward
            * termination?
    * 10,000 episodes
        * fixed across episodes: plaintext, key, ciphertext, 
        * whats fixed is the operations
        * point: is to perform the two actions

* RL
    * not remembering the history of env
    * is this not an issue in MDPs?
    * backtrack is the issue? how do you know?
        * how is backtracking implementing?
        * it is an action
* Neural Logic Machines, NLM
    * https://openreview.net/forum?id=B1xY-hRctX
* Problem
    * doesn’t learn the order
        * Or [x]
        * And [x]
        * XOR
    * check if it 1, 0.9, 0.2 
