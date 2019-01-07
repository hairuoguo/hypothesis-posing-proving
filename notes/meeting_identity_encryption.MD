

* encrypted string
* plain text using key to encrypted string
* Goal:
    * learn best sequence of actions
* how this connects to question asking
    * composition of primitives
* actions are not questions
* question, hypothesis, either true or false
    * e.g. "I think the way to get from the current string to the output string is an AND and OR"
    * question in lake sense: Q(Knowledge) = Ans
* RL
    * intermediate rewards, unclear
    * but the big issue is when the *final* reward in unambiguous
        * e.g. AlphaGo only ends when reaches end
        * note it has an advantage over maths or encryption in the sense Go always finishes so it always receives a positive or negative reward
            * to solve that we can put an upper bound on # of proof steps allowed or action to be taken
* what is a good question?
    * quantify it
    * ideas
        * lake paper on entropy


* hypothesis:
    * if we can build an AI system that asks
        * good questions
        * few questions
        * to obtain an answer/solve a task
* Action Plan
    * write RL code that solves/does some task, any. This is just to be familiar with RL code in the framework
        * do simple tutorial on TF or pytorch (depending which one you don’t know)
        * decide jointly which framework
        * then we RL tutorial
        * (by Wed)
    * identity environment prototype 0
        * have a set of actions, e.g. NOP, AND, OR, NOT
        * plaintext -> identity -> plaintext
        * can agent learn nothing changed?
        * what if we change only 1 bit? as encryption scheme?
        * (by friday)
    * reversible environment prototype 1
        * have a set of actions, e.g. NOP, AND, OR, NOT
            * perhaps add a few more? or agent should learn them by composition of stuff...
        * write a simple reversible encryption scheme (but not totally trivial)
        * what does agent do on it?
        * (need meeting to see if to be done)
