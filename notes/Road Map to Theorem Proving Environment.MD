https://docs.google.com/document/d/1ltK2ipX14CZhAYjWPggYJtkoqE-YNBTwH7RFuGfpeBM/edit

GOAL: capture the essence/principles
What "environment" looks like
Reward only at end
Possibly explore internal rewards
Some informational/content structure is revealed when hypotheses are verified
Increased structure isn't even weakly monotonic w.r.t. actual progress towards goal (but tends towards goal)
In expectation, verifying hypotheses brings us closer to the goal?
Prover has intuition/heuristic of closeness towards accomplishing goal
Uncertain of result of actions until actions are tried
Hypothesis verification/searching is costly
Come up with way to kill bad hypotheses early
Limit cost
Only continue with good hypotheses
what the actions would be there
Pose hypothesis
Actions towards verifying current hypothesis (manipulating current state)
Actions come from predefined grammar/universe of actions
what hypothesis would look like
Sub-result that would be useful for proof of entire theorem
E.g. "suppose sqrt(2) is rational" (proof by contradiction)
Must be 1. Verifiable/doable with actions and knowledge we have 2. Simpler than theorem itself
Making assumption (counterfactual), seeing if brings closer to goal (not sure how this fits in yet)
Main problems in theorem proving
Usefulness of hypotheses
How do we evaluate this? Probably by learning - this suggests having a “usefulness” distribution over the knowledge base
Balancing how informative a hypothesis is with solvability
Incorporating information learned from hypotheses
Composing/utilizing learned knowledge
simplify the problem/toy examples








Andy’s Outline of a theorem proving environment
Things the agent needs:
Need a growing differentiable knowledge base
Might want to preload this knowledge base with some axioms and proofs?
Have ability to split up problem into subproblems
Actions:
perform some atomic moves that are pre-defined (add, subtract, union of sets etc.)
Make a definition (this can include new atomic moves)
create a new symbol and make an assignment to it
Check for contradictions
Chunk the problem into pieces if examination of the environment state shows this to be possible
Use formal verification? (This can include simplifying the state to get rid of pieces that already have been proved)
Go into hypothesis/conjecture mode:
 make assumptions
Perform other actions
Verify
is counterfactual reasoning useful? (what if...?)
check with an example? (how useful is it to explicitly check examples to get intuition?)
----
The problems should be, know the problem complexity goes down we are getting closer to the solution
Goal: some examples that simplifies the problem, and write down the 3 principles it captures

