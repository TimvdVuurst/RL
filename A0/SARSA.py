#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        G = r + self.gamma * np.max(self.Q_sa[s_next,a_next])
        self.Q_sa[s,a] += self.learning_rate * (G - self.Q_sa[s,a])

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500, verbose = False):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    #
    s = env.reset() # Initial state
    a = pi.select_action(s,policy,epsilon,temp) # Sample action
    for episode in range(n_timesteps): 
        # Both 'primes' refer to 'next' here, prime is used to be in line with the given pseudocde

        s_prime, r, done = env.step(a) # Sample environment from that action
        a_prime = pi.select_action(s_prime,policy,epsilon,temp)
        pi.update(s,a,r,s_prime,a_prime, done) # Update the Qvalues

        if done: # If s' reached the goal, reset
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            s = s_prime
            a = a_prime

        if episode % eval_interval == 0:
            r_mean = pi.evaluate(eval_env)
            eval_returns.append(r_mean)
            eval_timesteps.append(episode)
            if verbose:
                print(f'EVALUATION at {episode = } yields mean return of {r_mean}')

        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 10000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = False
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, 100, True)
            
    
if __name__ == '__main__':
    test()
