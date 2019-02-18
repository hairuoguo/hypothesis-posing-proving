## Proposed Compositional Environment

# Desiderata:
    
* progressive reveal of information
* question-posing
* premise selection
* compositionality
* knowledge base
* lemma creation/synthesis
* can be combined with bit reversal environment

# Summary

This objective of this environment is similar to that of the bit reversal environment -- given a state bitstring and a target bitstring, as well as a set of sequences of actions (premises), select and compose the sequences of actions to find the shortest sequence between the state bitstring and the target bitstring.

Sequences of actions can be provided at the beginning, or they can be the resulting sequences of actions that are found for solving a particular bitstring from the bitstring reversal environment. That is, the agent must first solve n bitstring reversal subtasks, and then use the results from those tasks to solve this final task. Eventually, a goal to strive for would be to have the agent __learn to pose the bitstrings for the bitstring reversal subtasks itself__ in order to learn useful sequences of actions.

# Questions to answer:

How does occluded bits fit into this?

How do we incentivize the agent to pose bitstring reversal subtasks instead of tackling the larger problem itself? How do we get the agent to map the larger task to (simpler) subtasks?

* repeated patterns
* assumptions -> what would happen if something were the case

Can we find a mapping from this overall environment/problem to the Hamming Space (and Linear Algerbra at large)?

Is it possible to map this environment onto any other interesting tasks?
