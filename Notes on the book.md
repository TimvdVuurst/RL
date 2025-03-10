## Deep Q-learning
*"Deep reinforcement learning is based on the observation that bootstrapping is*
*also a kind of minimization process in which an error (or difference) is minimized"*

For convergence of algorithms such as Q-learning, the selection rule
must guarantee that eventually all states will be sampled by the environment. This cannot hold for large problems. A difference with the supervised learning is that in Q-learning subsequent
samples are not independent. The next action is determined by the current policy,
and will most likely be the best action of the state.

*Deep learning and Q-learning look similar in structure. Both
consist of a double loop in which a target is optimized, and we can wonder if
bootstrapping can be combined with loss-function minimization. This is indeed
the case, as Mnih et al. showed in 2013.*

The loss function for Q-learning is the squared difference between the output of the forward-pass $Q_{\theta_t}$ and the old (classical) update *target* : $r + \gamma \cdot \text{max}_{a'} \left(Q_{\theta_{t-1}}(s',a')\right)$. An important observation is that the update targets depend on the previous network weights 𝜃𝑡􀀀1 (the optimization targets move during optimization); this is in contrast with the targets used in a supervised learning process, that are fixed before learning begins.
There are three problems with this naive deep Q-learner. First, convergence to the
optimal Q-function depends on full coverage of the state space, yet the state space is
too large to sample fully. Second, there is a strong correlation between subsequent
training samples, with a real risk of local optima. Third, the loss function of gradient
descent literally has a moving target, and bootstrapping may diverge. 

### Deadly Triad
Function approximation may attribute values to states inaccurately. In contrast
to exact tabular methods, that are designed to identify individual states exactly,
neural networks are designed to individual features of states. Function approximation may thus cause mis-identification of states, and reward values and Q-values that are not assigned correctly.
Bootstrapping of values builds up new values on the basis of older values. Bootstrapping increases the efficiency of the training because values do not have to be calculated from the start. However, errors or biases in initial values may persist, and spill over to other states as values are
propagated incorrectly due to function approximation. Bootstrapping and function
approximation can thus increase divergence. Off-policy learning uses a behavior policy that is different from the target policy that we are optimizing for (Sect. 2.2.4.4). When the behavior policy is improved, the o-policy values may not improve. Off-policy learning converges generally less
well than on-policy learning as it converges independently from the behavior policy.
With function approximation convergence may be even slower, due to values being
assigned to incorrect states. 

The original focus of DQN is on breaking correlations between subsequent states,
and also on slowing down changes to parameters in the training process to improve
stability. The DQN algorithm has two methods to achieve this: **(1) experience replay**
and **(2) infrequent weight updates.** 