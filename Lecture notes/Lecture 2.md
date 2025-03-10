Reinforcement learning is learning by interaction. \
(state, action) $\rightarrow$ reward value.\
**Supervised** learning has a dataset. **Reinforcement** learning does not have any dataset.

**SDP**: Sequential Decision Problems.
In supervised learning this is a single step: $x \rightarrow y$. In RL this is multiple steps. **The Credit Assignment Problem**: How do we distribute the reward?

**MDP**: Markov Decision Problem. Reminiscent of Markov Chain. Future state is solely determined by the current state + action. Variables:
- S: State
- A: Action
- T: Transition probability
- R: Reward
- $\gamma$: Discount factor

**The Goal of RL** is to decide what action to take in any state. E.g.: *find the optimal policy such that in each state the action maximizes the expected cumulative future reward*:
$$
 \pi^*(s) \rightarrow a.
$$ 

**State**\
Uniquely represent the state of the environment at time $t$. E.g. location on a map, pieces on a board, angles of a joint, pixel values in a grid. 

**Transition**\
Basically: state s $\rightarrow$ action a $\rightarrow$ new state s'. $T_a(s,s')$ is the probability an action $a$ in state $s$ will transition to $s'$. The part s $\rightarrow$ a is chosen by the **agent/policy**. The part a $\rightarrow$ s is chosen by the **environment**. $T$ is known by the *environment* and not by the *agent.* Some transitions may be deterministic, such as in a puzzle or a grid-based game. 

**Action**\
Kinda speaks for itself. Can be discrete or continuous.

**Reward**
$r_a(s,s')$ is the reward received after action $a$ transitions from $s$ to $s'$. $r$ is the reward in state $s'$. RL aims to *maximize* reward (in constrast to minimizing cost/loss fuction).

**Discount factor**\
$\gamma$ discounts the importance of future rewards. This is especially important in continuous problems. In some *episodic* problems, it is left out (e.g. $\gamma = 1$).

**Episode** = trajectory = trace\
Episodic problems have an *end*. Continuous problem continue forever. An episode/trajectory/trace is the sequence of state/action/reward from start to finish:
$$
\tau_t^n = \{s_t,a_t,r_t,s+{t+1},a_{t+1},r_{t+1},...\}
$$

**Policy** - the most important word in RL\
Policy is the function of states to actions. $\pi(s) \rightarrow a$.

 - *Deterministic policy*: $\pi(s) \rightarrow a$
  
 - *Stochastic policy*: $\pi(a|s) \rightarrow$ probability distribution over actions.

**State value function:** $V$.

- $R(\tau)$ is the Return, the **cumulative** (discounted, future) reward of a trajectory:
  
$$
R(\tau) = \sum_{t=0}^T \gamma^t r_t
$$

- $V(s)$ is the **expected** return of the trajectory starting in state s and then following policy $\pi$; the expected cumulative discounted future reward aka the State Value Function:
  
$$
V^\pi(s) = E_{\tau \sim \pi}(R(\tau)|s_0 = s)
$$

**Q(s,a)**\
The expected return of taking action $a$ in state $s$ and then following the trajectory from $\pi$.

- $Q^{\pi}(s,a) = E_{\tau \sim \pi}(R(\tau)|s_0 = s, a_0=a)$

- $V^{\pi}(s) = E_{a\sim\tau}[Q^{\pi}(s,a)]$

- Optimal: $V^{*}(s) = \text{max}_{a}[Q^{*}(s,a)]$

**Policy improvement**\
Basically all algorithms follow: *Generalized Policy Improvement*:
- **Evaluate** the current policy, e.g., given a policy, find the Value of the policy for some states
  
- **Improve** current Policy; knowing these state/action values, insert better action for some states of the policy.

