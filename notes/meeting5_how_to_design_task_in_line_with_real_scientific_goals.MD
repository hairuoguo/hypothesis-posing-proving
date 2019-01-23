* design experiments that solve task
    * theorem proving
    * asking questions
* designing toy environment that capture aspects of that task in a simplified/easier setting
* episodes
    * random plaintext, key
    * constant ops
* try
    * 1
        * memorized
    * 0.9
        * memorized
    * 0.2
        * memorized
    * 0.1
        * sort of

* transfer learning
* not updating when there are mistakes
* how to incorporate structure into the environment
    * connect back to main Questions to determine what sort of structure to incorporate
        * theorem proving
        * question asking
* predicting between similarity between problems
    * can help curriculum learning
* predicting how much progress the agent is doing
* we definitively need structure for prediction
    * half same plantext
    * fix plaintext flip small part of it randomly
* question asking
    * card on her face, ask questions to guess character
    * poses a riddle
        * ask question, which I am allowed yes or no
    * connections to theorem proving
    * oracle that says yes or no
    * try to answer a mathematical problem
        * agent learns to pose questions/mathematical formulas
* learn from mistakes
    * repeat the examples that we got wrong

* tasks to be done
    * do experiments on learning from mistakes
        * come back to examples we get wrong
        * experiment with doing update or no update, as you want
            * an idea: if u can figure out why you did a mistake update, otherwise don't
    * thinking about how to define a task/environment such that we capture aspects of the problems we want to solve
        * theorem proving
            * how to deal with sparse rewards
                * idea: have a complexity/difficulty prediction (intuition) function that predicts how far from solving the task we probably are
        * asking question
            * inspiration from posing a problem and agent is only allowed to ask the oracle questions such that it receives 1/0 answers and using those it solves the problem
                * idea: to adapt it to bit string environment, the agent is given a set of questions its allowed to pose from some library/alphabet etc, composing those it forms questions that gives it information about the task it is currently solving and uses it to solve it
                    * THINK HERE!
