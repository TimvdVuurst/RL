**This week:**  First algorithms for solving problems; Tabular, that we can keep in memory.

## The Control Problem
- Find the optimal policy
	- For every state, the best action $\pi(s) \rightarrow a$ 

We will see three algorithms.

#policy 
But first: *Policy evaluation  / Improvement*. Policy is a function of states to actions.  

Begin with: policy evaluation. E.g. find the function $V^\pi (s)$ . If we have S/T/A/R/$\pi/\gamma$, then recursively calculate $V$. **Note: Bellman writes $P$ where Aske writes $T$**. This is the idea of *Dynamic Programming (DP)*: Divide and Conquer. 

Next: *Policy Iteration*. 
	Evaluate an episode with DP 
	Improve policy 
	Greedily repeat

Policy evaluation and iteration look so much alike, can we not put them together? Then we get:
## Value Iteration (VI)
Combine (interleave) policy evaluation & improvement in each timestep, and at the end recover the policy explicitly. It does the same thing as policy iteration, but is more efficient.  #seeslide 

For each state, for each action, find the best one is the basic idea. 

Recursively *improves* the policy using Bellman *evaluation*. **Positive**: smart interleaving of policy eval and improvement at each timestep. **Negative**; it needs the transition function $T$ (or $P$). But the agent generally does not have access to $T$! Womp womp. So we need:
## Monte Carlo
No access to the full transition function. So what do you do? You *sample* a full *episode*. 
**In:**
- Rewards
**Computes:**
- Optimal policy by sampling and averaging full episodes, assign Q value to all states in episode; interleaves policy eval and policy improvement by episode.
**Out:**
- Value of state-actions ($Q$)
- Policy of states ($\pi(s)\rightarrow a$)

#seeslide For pseudocode. 

Samples episodes from environment, no need for $T$. Generates episode, evaluates episode, improves policy after episode and repeats. 
**Positive**: 
Independent episodes: breaks bootstrapping, low bias.
**Negative**: 
High variance between episodes (slow convergence). 
## Temporal Difference (SARSA)
lost attention a little here so #seeslide 

SARSA vs Q-learning: only difference in on policy and off policy respectively. Empirically, Q-learning is *usually* better, but this is not formally proven. Therefore we learn both. 